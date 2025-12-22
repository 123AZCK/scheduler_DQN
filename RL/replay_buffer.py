import random
import numpy as np
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = []
        self.capacity = capacity
        self.device = device
        self.pos = 0

    def add(self,state, action, reward, next_state, done):
        if isinstance(state, torch.Tensor):
            state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state.cpu().numpy()
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.pos = (self.pos + 1)%self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states,actions,rewards,next_states,dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


