import os
import time
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from schedule_5_DQN.models.task import Task
from schedule_5_DQN.models.device import Device
from schedule_5_DQN.models.predictor import ExecutionTimePredictor
from schedule_5_DQN.Env.env import SchedulingEnv
from schedule_5_DQN.RL.scheduler import RLScheduler
def build_tasks():
    tasks = [
        Task(0, F=50, M=120, P=2, S=-0.2, dependencies=[],output_bytes=4000000),
        Task(1, F=320, M=180, P=8, S=0.8, dependencies=[0],output_bytes=300000),
        Task(2, F=650, M=260, P=16, S=0.95, dependencies=[0],output_bytes=80000),
        Task(3, F=420, M=240, P=8, S=0.85, dependencies=[1],output_bytes=5000),
        Task(4, F=80, M=90, P=2, S=0.3, dependencies=[2],output_bytes=2000),
        Task(5, F=120, M=200, P=4, S=0.2, dependencies=[2],output_bytes=600000),
        Task(6, F=60, M=100, P=2, S=-0.1, dependencies=[3, 4],output_bytes=50000),
        Task(7, F=500, M=140, P=16, S=1.0, dependencies=[5],output_bytes=100000),
        Task(8, F=410, M=150, P=8, S=-0.3, dependencies=[4,7],output_bytes=4000000),
        Task(9, F=100, M=100, P=2, S=0.1, dependencies=[6,8]),
    ]
    return tasks

def build_devices():
    devices = [
        # CPU
        Device(
            0,
            F_peak=180,
            B_peak=120,
            eff_comp=0.75,
            eff_mem=0.85,
            latency=0.005,
            net_bandwidth=3e9
        ),

        # GPU
        Device(
            1,
            F_peak=400,
            B_peak=250,
            eff_comp=0.9,
            eff_mem=0.92,
            latency=0.02,
            net_bandwidth=7e9
        ),

        # NPU / AI Accelerator
        Device(
            2,
            F_peak=550,
            B_peak=180,
            eff_comp=0.92,
            eff_mem=0.80,
            latency=0.03,
            net_bandwidth=5e9
        ),
    ]
    return devices


def train(num_episodes = 2000):
    num_tasks = 10
    num_devices = 3
    reward_history = []
    loss_history = []
    makespan_history = []
    env = SchedulingEnv(build_tasks(),build_devices(),ExecutionTimePredictor())

    state_dim = env._get_state().shape[0]
    agent = RLScheduler(state_dim=state_dim, num_tasks=num_tasks, num_devices=num_devices,
                        hidden_dim=128, lr=1e-3, buffer_size=20000, batch_size=64, target_update=200)
    results = []
    for ep in range(1, num_episodes + 1):
        env = SchedulingEnv(build_tasks(), build_devices(), ExecutionTimePredictor())
        state = env.reset()
        total_reward = 0.0
        losses = []
        steps = 0
        # keep stepping until done
        while True:
            ready = env.get_ready_tasks()
            avail = env.get_available_devices()
            task_id, device_id = agent.select_action(state, ready, avail)
            if task_id is None:
                # cannot schedule -> advance until next completion
                finished = env.auto_advance()
                state = env._get_state()
                if env.get_ready_tasks() == [] and finished == []:
                    # safety break
                    break
                continue

            next_state, reward, done, info = env.step((task_id, device_id))
            agent.store(state, task_id, device_id, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            total_reward += reward
            agent.update_epsilon()
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break
        reward_history.append(total_reward)
        loss_history.append(np.mean(losses) if len(losses) > 0 else None)
        makespan_history.append(env.current_time)
        results.append(total_reward)
        if ep % 200 == 0:
            avg = np.mean(results[-10:])
            print(f"EP {ep} total_reward {total_reward:.3f} avg10 {avg:.3f} eps {agent.epsilon:.3f}")

        # save model
    torch.save(agent.policy_net.state_dict(), "results/reward_2/dqn_scheduler.pth")
    print("Training finished, model saved to dqn_scheduler.pth")
    plot_training_curves(reward_history, loss_history, makespan_history)
    return reward_history

def plot_training_curves(rewards, losses, makespan):
    """绘制 Reward / Loss / Makespan 曲线"""
    plt.figure(figsize=(8, 10))

    # --- Reward ---
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()

    # --- Loss ---
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title("Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid()

    # --- Makespan ---
    plt.subplot(3, 1, 3)
    plt.plot(makespan)
    plt.title("Makespan per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Time")
    plt.grid()

    plt.tight_layout()
    plt.savefig("training_1.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_mean_std_shaded(runs_values, label="Reward", xlabel="Episode",
                         save_path=None, figsize=(10,5)):
    arr = np.asarray(runs_values)
    if arr.ndim != 2:
        raise ValueError("runs_values 必须是二维 (num_runs, num_steps)")

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    x = np.arange(arr.shape[1])

    plt.figure(figsize=figsize)
    plt.plot(x, mean, label=label, linewidth=1.5)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(label)
    plt.title(f"{label} (Mean ± Std)")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
def plot_confidence_interval(runs_values, label="Reward", xlabel="Episode",
                             confidence=0.95, save_path=None, figsize=(10,5)):
    arr = np.asarray(runs_values)
    num_runs = arr.shape[0]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1)

    # 对应正态分布的 Z 值
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.90:
        z = 1.645
    elif confidence == 0.99:
        z = 2.576

    ci = z * (std / np.sqrt(num_runs))
    x = np.arange(arr.shape[1])

    plt.figure(figsize=figsize)
    plt.plot(x, mean, label=f"{label} Mean")
    plt.fill_between(x, mean - ci, mean + ci, alpha=0.25, label=f"{int(confidence*100)}% CI")
    plt.xlabel(xlabel)
    plt.ylabel(label)
    plt.title(f"{label} with {int(confidence*100)}% Confidence Interval")
    plt.grid(linestyle="--", alpha=0.5)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
if __name__ == "__main__":
    train(200)
    # all_rewards = []  # shape -> (num_runs, num_steps)
    #
    # for i in range(10):
    #     rewards = train(400)
    #     all_rewards.append(rewards)
    #
    # all_rewards = np.array(all_rewards)
    # plot_mean_std_shaded(all_rewards, label="Reward", save_path="reward_shaded.png")
    # #plot_confidence_interval(all_rewards, label="Reward", confidence=0.95, save_path="reward_confidence.png")



