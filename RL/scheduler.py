import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from torch.nn import SmoothL1Loss

from .network import QNetwork
from .replay_buffer import ReplayBuffer

class RLScheduler:
    def __init__(self, state_dim, num_tasks, num_devices, hidden_dim=128,
                 lr=0.001, gamma=0.99,epsilon=1.0,epsilon_end=0.01,epsilon_decay=0.95,
                 buffer_size=50000,batch_size=64,target_update=500,device=None):
        self.state_dim = state_dim
        self.num_tasks = num_tasks
        self.num_devices = num_devices
        self.action_dim = num_devices * num_tasks
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim, hidden_dim,self.action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, hidden_dim,self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optim = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.replay = ReplayBuffer(buffer_size,device=self.device)
        self.train_steps = 0
        self.loss_fn = SmoothL1Loss()

    def select_action(self, state, ready_ids, available_ids):
        if not ready_ids or not available_ids:
            return None, None

        if random.random()<self.epsilon:
            task_idx = random.choice(ready_ids)
            device_idx = random.choice(available_ids)
            return task_idx, device_idx

        # evaluate Qs
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state_t = state.to(self.device).unsqueeze(0)

        with torch.no_grad():
            qvals = self.policy_net(state_t).squeeze(0).cpu().numpy()

        best_v = -1e2
        best_action = (None, None)
        for t in ready_ids:
            for d in available_ids:
                action_id = int(t * self.num_devices + d)
                if action_id > len(qvals):
                    continue
                v = qvals[action_id]
                if v > best_v:
                    best_v = v
                    best_action = (t, d)

        return best_action

    def encode_action(self, task_id, device_id):
        return int(task_id * self.num_devices + device_id)

    def decode_action(self, action_index):
        t = action_index // self.num_devices
        d = action_index % self.num_devices
        return t, d

    def store(self, state, task_id, device_id, reward, next_state, done):
        action = self.encode_action(task_id, device_id)
        self.replay.add(state, action, reward, next_state, float(done))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_step(self):
        batch = self.replay.sample(self.batch_size)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = batch

        q_values = self.policy_net(states)#当前状态下所有动作的Q值，[[1,2,3][2,3,4],[1,3,5]]
        q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        #取出对应动作的Q值
    #double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)

            next_q_values = self.target_net(next_states)
            next_q = next_q_values.gather(1, next_actions).squeeze()
            #next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_action, target_q)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()




