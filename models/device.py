class Device:
    """
        设备类 - 表示异构计算平台中的计算设备（如CPU、GPU等）

        属性说明:
        - id: 设备唯一标识符
        - F_peak: 峰值计算能力 (单位：FLOPS，每秒浮点运算次数)
        - B_peak: 峰值内存带宽 (单位：bytes/s，每秒字节数)
        - latency: 基础延迟 (设备启动任务的基础时间开销)
        - eff_comp: 计算效率因子 (0-1，实际计算能力与峰值计算能力的比例)
        - eff_mem: 内存效率因子 (0-1，实际内存带宽与峰值内存带宽的比例)
        - busy_until: 设备忙碌直到的时间点
        - current_task: 当前正在运行的任务
        - completed_tasks: 已完成的的任务列表
        - total_busy_time: 设备总忙碌时间
    """
    def __init__(self, did, F_peak, B_peak, eff_comp=0.7,
                 eff_mem =0.7,latency=1e-4,net_bandwidth=1e9):
        self.id = did
        self.F_peak = F_peak
        self.B_peak = B_peak
        self.eff_comp = eff_comp
        self.eff_mem = eff_mem
        self.latency = latency
        self.net_bandwidth = net_bandwidth
        self.busy_until = 0.0
        self.total_busy_time = 0.0
        self.current_task = None
        self.completed_tasks = []

    def is_available(self, current_time):
        return current_time >= self.busy_until

    def assign_task(self,task_id, start_time, exec_time):
        self.current_task = task_id
        self.busy_until = start_time + exec_time
        self.total_busy_time += exec_time

    def complete_task(self):
        if self.current_task is not None:
            self.current_task = None
            self.completed_tasks.append(self.current_task)

    def get_utilization(self, total_time):
        if total_time <= 0:
            return 0.0
        return min(1.0, self.total_busy_time / max(1e-12, float(total_time)))

    def reset(self):
        self.current_task = None
        self.completed_tasks = []
        self.busy_until = 0.0
        self.total_busy_time = 0.0

    def __repr__(self):
        return (f"Device(id={self.id}, F={self.F_peak:.2e}, B={self.B_peak:.2e}, "
                f"eff_comp={self.eff_comp:.2f}, eff_mem={self.eff_mem:.2f}, busy_until={self.busy_until:.3f})")


if __name__ == "__main__":
    d = Device("cpu", F_peak=10e9, B_peak=50e9, eff_comp=0.8, eff_mem=0.8, latency=0.001)
    print(d)
    print("available at t=0?", d.is_available(0))
    d.assign_task(task_id=1, start_time=0.0, exec_time=0.5)
    print("after assign:", d)
    print("available at t=0.1?", d.is_available(0.1))
    print("available at t=0.6?", d.is_available(0.6))
    d.complete_task()
    print("after complete:", d)