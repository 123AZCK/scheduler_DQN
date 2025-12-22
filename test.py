import torch
import os
import numpy as np
from schedule_5_DQN.models.task import Task
from schedule_5_DQN.models.device import Device
from schedule_5_DQN.models.predictor import ExecutionTimePredictor
from schedule_5_DQN.Env.env import SchedulingEnv
from schedule_5_DQN.RL.scheduler import RLScheduler
from gantt import draw_gantt, draw_gantt_animation, draw_gantt_time_animation, draw_gantt_pred


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

def load_agent(model_path, env):
    """用环境尺寸创建 agent 并加载模型权重"""
    state_dim = env._get_state().shape[0]
    num_tasks = len(env.tasks)
    num_devices = len(env.devices)

    agent = RLScheduler(state_dim=state_dim, num_tasks=num_tasks, num_devices=num_devices)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    # 加载权重
    ckpt = torch.load(model_path, map_location="cpu")
    try:
        agent.policy_net.load_state_dict(ckpt)
    except Exception:
        # 如果 checkpoint 是 dict 形式（例如保存了 state_dict 键），尝试兼容
        if isinstance(ckpt, dict) and 'policy_net_state_dict' in ckpt:
            agent.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
        else:
            # 直接尝试把 ckpt 当作 state_dict
            agent.policy_net.load_state_dict(ckpt)
    agent.policy_net.eval()
    # 评估时禁用探索
    agent.epsilon = 0.0
    print(f"\nLoaded model from {model_path}. Agent epsilon set to 0 (greedy).")
    return agent


def pretty_info(info):
    parts = []
    for k in ('predicted', 'pred_exec_time', 'predicted_time', 'pred_time'):
        if k in info:
            parts.append(f"pred={info[k]:.4f}")
            break
    for k in ('actual', 'actual_exec_time', 'actual_time'):
        if k in info:
            parts.append(f"actual={info[k]:.4f}")
            break
    if 'reward' in info:
        parts.append(f"reward={info['reward']:.4f}")
    # include finished / completed info if present
    if 'finished' in info and info['finished']:
        parts.append(f"finished={info['finished']}")
    if 'completed_now' in info and info['completed_now']:
        parts.append(f"completed_now={info['completed_now']}")
    return ", ".join(parts) if parts else str(info)


def run_test(predictor_update_on_step, model_path="results/reward_2/dqn_scheduler.pt"):
    # 构建 env（与训练时一样）
    tasks = build_tasks()
    devices = build_devices()
    predictor = ExecutionTimePredictor()
    env = SchedulingEnv(tasks, devices, predictor, predictor_update_on_step,noise=0.05)
    state = env.reset()
    task_records = []
    # 加载 agent
    agent = load_agent(model_path, env)

    print("\n=== Start test episode (greedy policy) ===\n")
    step = 0
    # 主循环：直到 done
    while True:
        step += 1
        ready = env.get_ready_tasks()
        avail = env.get_available_devices()

        # 若无可调度任务，则推进时间到下一个事件
        if (not ready) or (not avail):
            finished = env.auto_advance() if hasattr(env, "auto_advance") else []
            if finished:
                print(f"[time {env.current_time:.4f}] auto_advance finished tasks: {finished}")
            # 如果既没有 ready 也没有 future events（env 可能返回 []），检查是否 done
            if not ready and not avail and (not finished):
                # 如果 env 有 done 检查接口则使用它，否则根据 completed 数量判断
                done_flag = False
                if hasattr(env, "_is_done"):
                    done_flag = env._is_done()
                else:
                    done_flag = (len(getattr(env, "completed", [])) == len(env.tasks))
                if done_flag:
                    break
                # 否则继续循环
            # 更新 state 变量
            state = env._get_state()
            continue

        # 使用 agent 贪心选择动作
        t_id, d_id = agent.select_action(state, ready, avail)
        if t_id is None:
            # agent 返回 None 表示没有动作，推进时间
            finished = env.auto_advance() if hasattr(env, "auto_advance") else []
            if finished:
                print(f"[time {env.current_time:.4f}] auto_advance finished tasks: {finished}")
            state = env._get_state()
            continue

        print(f"[time {env.current_time:.4f}] → schedule: task {t_id} -> device {d_id}")
        # 执行动作：env.step 接受 (task_idx, device_idx) 或 action_index（视你的 env 实现）
        # 我们以 tuple 形式调用（兼容大多数实现）
        next_state, reward, done, info = env.step((t_id, d_id))

        # 打印 info（兼容多种 key）
        if isinstance(info, dict):
            # 有些 env 返回 reward 在 info 中，有些不
            info_print = pretty_info(info)
        else:
            info_print = str(info)

        print(f"  executed: {info_print}  (reward={reward:.4f})")
        # 打印 any finished inside info
        if isinstance(info, dict) and ('finished' in info and info['finished']):
            print(f" finished tasks: {info['finished']}")
        if info["predicted"] > 0:
            task_records.append({
                "task": t_id,
                "device": d_id,
                "start": info["start_time"],
                "predicted": info["predicted"],
                "actual": info["actual"],
            })

        state = next_state

        if done:
            print(f"\nAll tasks finished at time {env.current_time:.4f} (makespan).")
            break

    print("\n=== Test finished ===\n")
    print(f"Final makespan: {env.current_time:.4f}")
    print("=" * 50)
    return task_records

if __name__ == "__main__":
    predictor_update_on_step = True
    predictor_update_on_step_pred = False
    rec_pred = run_test(predictor_update_on_step_pred, "results/reward_2/dqn_scheduler.pth")
    rec_actual = run_test(predictor_update_on_step, "results/reward_2/dqn_scheduler.pth")
    draw_gantt(rec_actual, len(build_devices()), save_path="results/reward_2/gantt_actual.png")
    draw_gantt_pred(rec_pred, len(build_devices()), save_path="results/reward_2/gantt_pred.png")
    #draw_gantt_animation(rec_pred, len(build_devices()), save_path="schedule.gif")
    #draw_gantt_time_animation(rec_pred, len(build_devices()), save_path="schedule_time.gif")



