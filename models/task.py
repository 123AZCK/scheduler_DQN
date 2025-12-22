from typing import List, Optional

class Task:
    """
        任务类 - 表示需要被调度的计算任务

        属性说明:
        - id: 任务唯一标识符
        - F: 计算量 (单位：浮点运算次数 FLOPs)
        - M: 内存访问量 (单位：字节 bytes)
        - P: 并行度因子 (大于1表示任务可以并行执行)
        - S: 特殊性因子 (0-1，表示任务对特定硬件的偏好程度)
        - dependencies: 当前任务的依赖任务
        - signature: 任务签名，用于唯一识别任务类型
        - assigned: 是否已被分配到设备
        - start: 任务开始执行时间
        - finish: 任务完成时间
        - completed: 任务是否已完成
    """
    STATE_PENDING = 0
    STATE_READY = 1
    STATE_RUNNING = 2
    STATE_DONE = 3
    def __init__(self, tid, F, M, P=1.0, S=0.0, dependencies: Optional[List[int]] = None,
                 output_bytes=0,signature: Optional[str] = None):
        self.id = tid
        self.F = F
        self.M = M
        self.P = P
        self.S = S
        self.dependencies = list(dependencies) if dependencies else []
        self.output_bytes = output_bytes
        self.signature = signature if signature is not None else str(tid)
        self.data_location = None
        self.children = []
        self.start_time = None
        self.finish_time = None
        self.state = Task.STATE_PENDING
        self.assigned = False
        self.arrival_time = 0.0
        self.depth = 0

    def is_ready(self, done_set):
        return all(d in done_set for d in self.dependencies)

    def mark_ready(self):
        self.state = Task.STATE_READY
        self.assigned = False

    def mark_running(self, start_time: float):
        self.state = Task.STATE_RUNNING
        self.assigned = True
        self.start_time = float(start_time)

    def mark_done(self, finish_time: float, device_id):
        self.state = Task.STATE_DONE
        self.assigned = False
        self.finish_time = float(finish_time)
        self.data_location = device_id

    def reset(self):
        self.state = Task.STATE_PENDING
        self.assigned = False
        self.start_time = None
        self.finish_time = None
        self.depth = 0

    def add_child(self, child_id: int):
        if child_id not in self.children:
            self.children.append(child_id)

    def __repr__(self):
        return (f"Task(id={self.id}, F={self.F:.2e}, M={self.M:.2e}, P={self.P}, S={self.S}, "
                f"state={self.state}, deps={self.dependencies}, children={self.children})")


if __name__ == "__main__":
    t = Task(1, F=1e9, M=1e6, P=2.0, S=0.5, dependencies=[0,1,2,3])
    print("Task:", t)
    print("is_ready :", t.is_ready({0,1}))
    print("is_ready ( completed):", t.is_ready({0,1,2,3}))
    t.mark_ready()
    print("after mark_ready:", t)
    t.mark_running(start_time=0.1)
    print("after running:", t)
    t.mark_done(finish_time=1.23,device_id=0)
    print("after done:", t)
