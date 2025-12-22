import numpy as np
from typing import List, Dict, Any
from schedule_5_DQN.models.task import Task


class SchedulingEnv:


    def __init__(self, tasks, devices, predictor, predictor_update_on_step=True,noise=0.1):
        self.tasks = tasks
        self.devices = devices
        self.predictor = predictor
        self.noise = noise

        self.num_tasks = len(tasks)
        self.num_devices = len(devices)

        self.current_time = 0.0
        self.completed = set()

        self.in_degree = {t.id: len(t.dependencies) for t in self.tasks}
        self.dependents = {t.id: [] for t in self.tasks}
        for t in self.tasks:
            for deps in t.dependencies:
                self.dependents[deps].append(t.id)

        self.task_map = {t.id: t for t in self.tasks}
        self.device_map = {d.id: d for d in self.devices}
        self.predictor_update_on_step = predictor_update_on_step

    def reset(self):
        self.in_degree = {t.id: len(t.dependencies) for t in self.tasks}

        self.dependents = {t.id: [] for t in self.tasks}
        for t in self.tasks:
            for deps in t.dependencies:
                self.dependents[deps].append(t.id)

        self.current_time = 0.0
        self.completed = set()

        for t in self.tasks:
            t.reset()
        for d in self.devices:
            d.reset()

        return self._get_state()


    def _get_state(self):
        maxF = max((t.F for t in self.tasks), default=1.0)
        maxM = max((t.M for t in self.tasks), default=1.0)
        max_device_time = max((d.busy_until for d in self.devices), default=1.0)
        if max_device_time <= 0:
            max_device_time = 1.0

        task_state = []
        for t in self.tasks:
            completed = 1 if t.id in self.completed else 0
            running = 1 if t.state == Task.STATE_RUNNING else 0
            ready = 1 if (self.in_degree[t.id] == 0
                          and t.id not in self.completed
                          and t.state != Task.STATE_RUNNING) else 0

            task_state.extend([
                float(t.F) / float(maxF),
                float(t.M) / float(maxM),
                float(t.P) / max(1.0, float(t.P)),
                float(t.S),
                completed,
                ready,
                running
            ])

        device_state = []
        for d in self.devices:
            is_busy = 1 if d.busy_until > self.current_time else 0
            time_left = max(0.0, d.busy_until - self.current_time) / max_device_time
            device_state.extend([is_busy, time_left])

        return np.array(task_state + device_state, dtype=np.float32)


    def get_ready_tasks(self):
        ready = []
        for t in self.tasks:
            if t.id in self.completed:
                continue
            if self.in_degree[t.id] != 0:
                continue
            if t.state == Task.STATE_RUNNING:
                continue
            ready.append(t.id)
        return ready

    def get_available_devices(self):
        return [d.id for d in self.devices if d.is_available(self.current_time)]

    def get_transfer_time(self, task, target_device):
        pred_tasks_id = task.dependencies
        if not pred_tasks_id:
            return 0.0
        transfer_end_time = []
        for t_id in pred_tasks_id:
            p_task = self.task_map[t_id]
            src_device = self.devices[p_task.data_location]
            if src_device == target_device:
                start_time = 0.0 + src_device.busy_until
                transfer_end_time.append(start_time)
                continue
            else:
                time = p_task.output_bytes/min(src_device.net_bandwidth, target_device.net_bandwidth)
                start_t_time = time + src_device.busy_until
                transfer_end_time.append(start_t_time)

        return max(transfer_end_time)



    def step(self, action):
        """
        action 是 (task_id, device_id)
        """

        task_id, device_id = action
        info = {'task_id': task_id, 'device_id': device_id}

        # 无效输入
        if task_id not in self.task_map:
            return self._get_state(), -1.0, False, {**info, 'error': 'invalid_task'}
        if device_id not in self.device_map:
            return self._get_state(), -1.0, False, {**info, 'error': 'invalid_device'}

        task = self.task_map[task_id]
        device = self.device_map[device_id]

        # 必须 ready
        if task_id in self.completed \
                or self.in_degree[task_id] != 0 \
                or task.state == Task.STATE_RUNNING:
            return self._get_state(), -0.5, False, {**info, 'error': 'task_not_ready'}

        # 设备必须可用
        if not device.is_available(self.current_time):
            return self._get_state(), -0.5, False, {**info, 'error': 'device_not_available'}

        old_ms = max(d.busy_until for d in self.devices)
        pred_time = self.predictor.predict_execution_time(task, device)
        noise_factor = np.random.uniform(1 - self.noise, 1 + self.noise)
        actual_time = pred_time * noise_factor
        transfer_end_time = self.get_transfer_time(task, device)
        info['predicted'] = pred_time
        info['actual'] = actual_time
        info['noise'] = noise_factor

        start_time = max(self.current_time, device.busy_until,transfer_end_time)
        if self.predictor_update_on_step:
            device.assign_task(task_id, start_time, actual_time)
        else:
            device.assign_task(task_id, start_time, pred_time)
        task.mark_running(start_time)
        info['start_time'] = start_time
        waiting_time = max(0.0, start_time - self.current_time)
        new_ms = max(d.busy_until for d in self.devices)

        # 预测时间后，直接更新预测模型？不太符合逻辑，也许应该全部调度完之后（12.5，不需要，任务与设备一一对应更新），再更新模型，同时test时感觉不需要这一步
        #12.8训练时模拟出的真实运行时间，实际上硬件时该怎么办
        if self.predictor_update_on_step:
            reward = - (waiting_time + actual_time)
            self.predictor.update_model(task, device, actual_time, pred_time)
        else:
            reward = - (waiting_time + pred_time)

        reward += 2 * (old_ms - new_ms)
        finished = self.auto_advance()
        info["finished"] = finished

        done = len(self.completed) == self.num_tasks

        return self._get_state(), reward, done, info


    def auto_advance(self):

        ready = self.get_ready_tasks()
        available = self.get_available_devices()

        # 如果 ready 不为空，且有设备可用，则不推进时间
        if len(ready) > 0 and len(available) > 0:
            return []

        # 计算下一个事件时间
        future_times = [d.busy_until for d in self.devices if d.busy_until > self.current_time]

        if not future_times:
            return []  # nothing running

        next_time = min(future_times)

        # 时间推进
        self.current_time = float(next_time)

        # 检查完成的任务
        return self._process_completions()

    def _process_completions(self):
        finished = []
        for d in self.devices:
            t_id = d.current_task
            if t_id is None:
                continue
            if d.busy_until <= self.current_time + 1e-12:
                t = self.task_map[t_id]
                t.mark_done(d.busy_until,d.id)
                self.completed.add(t_id)
                finished.append(t_id)

                # 降低后继任务的 in_degree
                for child in self.dependents[t_id]:
                    self.in_degree[child] -= 1

                d.complete_task()

        return finished


    def render(self):
        print(
            f"[time {self.current_time:.3f}] ready={self.get_ready_tasks()} "
            f"available={self.get_available_devices()} "
            f"completed={len(self.completed)}/{self.num_tasks}"
        )
