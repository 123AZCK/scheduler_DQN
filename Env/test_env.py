from schedule_5_DQN.models.task import Task
from schedule_5_DQN.models.device import Device
from schedule_5_DQN.models.predictor import ExecutionTimePredictor
from schedule_5_DQN.Env.env import SchedulingEnv
import random
import numpy as np
# build tasks (manual DAG)
tasks = [
    Task(0, F=100, M=50, P=1, S=0.0, dependencies=[]),
    Task(1, F=120, M=30, P=2, S=0.2, dependencies=[0]),
    Task(2, F=90, M=80, P=1, S=0.1, dependencies=[0]),
    Task(3, F=156, M=40, P=3, S=0.3, dependencies=[1, 2]),
    Task(4, F=155, M=40, P=1, S=0.2, dependencies=[2]),
    Task(5, F=120, M=20, P=2, S=0.1, dependencies=[0, 1]),
    Task(6, F=170, M=50, P=3, S=0.0, dependencies=[3, 4, 5]),
    Task(7, F=444, M=80, P=2, S=1.0, dependencies=[6, 0]),
    Task(8, F=144, M=10, P=1, S=0.0, dependencies=[6, 5, 4]),
    Task(9, F=66, M=10, P=1, S=0.1, dependencies=[7, 8]),
]

# devices (ensure .id are integers or strings that match device_map usage)
devices = [
    Device(0, F_peak=200, B_peak=100, eff_comp=0.8, eff_mem=0.8, latency=0.01),
    Device(1, F_peak=500, B_peak=200, eff_comp=0.9, eff_mem=0.9, latency=0.02),
    Device(2, F_peak=300, B_peak=80,  eff_comp=0.85, eff_mem=0.75, latency=0.05),
]

pred = ExecutionTimePredictor()

env = SchedulingEnv(tasks, devices, pred, noise=0.05)
state = env.reset()

done = False
step_id = 0

while not done:
    env.render()
    ready = env.get_ready_tasks()
    avail = env.get_available_devices()

    # 若可以调度 → 调度
    if len(ready) > 0 and len(avail) > 0:
        t = ready[random.randint(0,len(ready)-1)]
        d = avail[random.randint(0,len(avail)-1)]
        print(f"\n→ 调度: task {t} 到 device {d}")
        s, r, done, info = env.step((t, d))
        print(f"  预测时间={info['predicted']:.4f}, 实际={info['actual']:.4f}, 奖励={r:.4f}")

    else:
        # 无法调度 → 自动推进到下一个事件
        finished = env.auto_advance()
        if finished:
            print(f"\n推进时间到了 {env.current_time:.4f}, 完成任务: {finished}")

    step_id += 1
    if step_id > 100:
        print("发生死循环，请检查环境逻辑！")
        break

env.render()
print("\n===== 测试结束 =====")
