import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import random


def draw_gantt_time_animation(task_records, num_devices, save_path="gantt_time.gif", fps=15):

    plt.style.use("seaborn-v0_8-whitegrid")

    max_time = max(r["start"] + r["predicted"] for r in task_records)

    fig, ax = plt.subplots(figsize=(4, 6))

    # 明亮颜色
    bright = [
        "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
        "#f1c40f", "#1abc9c", "#e67e22", "#34495e"
    ]
    task_colors = {r["task"]: bright[r["task"] % len(bright)] + "CC" for r in task_records}

    # 每帧推进的时间
    dt = max_time / 150    # 总共 150 帧，可调节
    total_frames = int(max_time / dt) + 2

    # 初始化背景
    def init_background():
        ax.clear()
        for dev in range(num_devices):
            ax.add_patch(
                patches.Rectangle((dev, 0), 1, max_time,
                                  facecolor="#fafafa", alpha=0.6)
            )
            ax.text(dev + 0.5, -0.25,
                    f"Device {dev}",
                    ha="center", fontsize=12, fontweight="bold")

            ax.axvline(dev, linestyle="--", color="gray", linewidth=0.5)

        ax.axvline(num_devices, linestyle="--", color="gray", linewidth=0.5)

        ax.set_xlim(0, num_devices)
        ax.set_ylim(0, max_time)
        ax.invert_yaxis()  # 时间正方向往下
        ax.set_xticks([])

        ax.set_ylabel("Time")
        ax.set_xlabel("Devices")

    # 每帧
    def update(frame):
        current_time = frame * dt
        init_background()

        for rec in task_records:
            task = rec["task"]
            dev = rec["device"]
            st = rec["start"]
            dr = rec["predicted"]
            color = task_colors[task]

            # 还没开始
            if current_time < st:
                continue

            # 已经结束 → 显示完整任务
            if current_time >= st+dr:
                height = dr
            else:
                # 正在执行 → 显示部分
                height = current_time - st

            rect = patches.Rectangle(
                (dev + 0.15, st),    # x,y
                0.7,                 # width
                height,              # height
                edgecolor="black",
                facecolor=color,
                linewidth=1,
                alpha=0.9
            )
            ax.add_patch(rect)

            # 只在任务全部执行后显示文字
            if current_time >= st+dr:
                ax.text(dev + 0.5, st + dr / 2,
                        f"T{task+1}",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold")

        ax.set_title(f"Time = {current_time:.2f} / {max_time:.2f}")

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, repeat=False
    )

    ani.save(save_path, writer="pillow", fps=fps)

    print(f"真实时间推进 GIF 动画已保存到：{save_path}")

    plt.close()


def draw_gantt_animation(task_records, num_devices, save_path="gantt.gif"):

    plt.style.use("seaborn-v0_8-whitegrid")

    max_time = max(r["start"] + r["predicted"] for r in task_records)

    fig, ax = plt.subplots(figsize=(4, 6))

    # 明亮配色
    bright = [
        "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
        "#f1c40f", "#1abc9c", "#e67e22", "#34495e"
    ]
    task_colors = {r["task"]: bright[r["task"] % len(bright)] + "CC" for r in task_records}

    # 初始化背景（只画一次）
    def init_bg():
        ax.clear()
        for dev in range(num_devices):
            ax.add_patch(
                patches.Rectangle((dev, 0), 1, max_time,
                                  color="#fafafa", alpha=0.6)
            )
            ax.text(dev + 0.5, -0.25, f"Device {dev}",
                    ha="center", fontsize=12, fontweight="bold")
            ax.axvline(dev, linestyle="--", color="gray", linewidth=0.6)
        ax.axvline(num_devices, linestyle="--", color="gray", linewidth=0.6)
        ax.set_xlim(0, num_devices)
        ax.set_ylim(0, max_time)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_ylabel("Time")
        ax.set_xlabel("Devices")

    # 每帧画到第 i 个任务
    def update(frame):
        init_bg()
        for rec in task_records[: frame + 1]:
            task = rec["task"]
            dev = rec["device"]
            start = rec["start"]
            duration = rec["predicted"]
            color = task_colors[task]

            rect = patches.Rectangle(
                (dev + 0.15, start), 0.7, duration,
                linewidth=1, edgecolor="black", facecolor=color, alpha=0.9
            )
            ax.add_patch(rect)
            ax.text(dev + 0.5, start + duration / 2,
                    f"T{task+1}", ha="center", va="center",
                    fontsize=11, fontweight="bold")

        ax.set_title(f"Scheduling Progress – {frame+1}/{len(task_records)} tasks")

    ani = animation.FuncAnimation(
        fig, update, frames=len(task_records), interval=600, repeat=False
    )

    # 保存 GIF
    ani.save(save_path, writer="pillow", fps=2)
    print(f"GIF 已保存: {save_path}")

    plt.close()
def draw_gantt(task_records, num_devices, save_path="gantt.png"):


    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(4, 6))

    random.seed(1)
    bright = [
        "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
        "#f1c40f", "#1abc9c", "#e67e22", "#34495e"
    ]
    task_colors = {r["task"]: bright[r["task"] % len(bright)] + "CC" for r in task_records}

    for dev in range(num_devices):
        ax.add_patch(
            patches.Rectangle(
                (dev, 0), 1, max((r["start"]+r["actual"]) for r in task_records),
                facecolor="#f8f8f8", edgecolor="none", alpha=0.5
            )
        )
        ax.text(dev+0.5, -0.3, f"Device {dev}",
                ha="center", va="top", fontsize=12, fontweight="bold")

        ax.axvline(dev, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(num_devices, color="black", linestyle="--", linewidth=0.5)


    for rec in task_records:
        task = rec["task"]
        dev = rec["device"]
        start = rec["start"]
        duration = rec["actual"]

        color = task_colors[task]

        rect = patches.Rectangle(
            (dev + 0.15, start),
            0.7, duration,
            linewidth=1.2,
            edgecolor="black",
            facecolor=color,
            alpha=0.85
        )
        ax.add_patch(rect)

        # 任务号
        ax.text(
            dev + 0.5,
            start + duration / 2,
            f"T{task+1}",
            ha="center",
            va="center",
            fontsize=12,
            color="black"
        )

    # --- 时间轴设置 ---
    ax.set_ylabel("Time", fontsize=12)
    ax.set_xlabel("Devices", fontsize=12)
    ax.set_xlim(0, num_devices)
    ax.set_ylim(0, max((r["start"]+r["actual"]) for r in task_records))
    ax.invert_yaxis()  # 时间向下

    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()

    # 高清保存
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"甘特图(实际)已保存到: {save_path}")

    plt.show()
def draw_gantt_pred(task_records, num_devices, save_path="gantt_pred.png"):


    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(4, 6))

    bright = [
        "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
        "#f1c40f", "#1abc9c", "#e67e22", "#34495e"
    ]
    task_colors = {r["task"]: bright[r["task"] % len(bright)] + "CC" for r in task_records}

    for dev in range(num_devices):
        ax.add_patch(
            patches.Rectangle(
                (dev, 0), 1, max((r["start"]+r["predicted"]) for r in task_records),
                facecolor="#f8f8f8", edgecolor="none", alpha=0.5
            )
        )
        ax.text(dev+0.5, -0.3, f"Device {dev}",
                ha="center", va="top", fontsize=12, fontweight="bold")

        ax.axvline(dev, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(num_devices, color="black", linestyle="--", linewidth=0.5)


    for rec in task_records:
        task = rec["task"]
        dev = rec["device"]
        start = rec["start"]
        duration = rec["predicted"]

        color = task_colors[task]

        rect = patches.Rectangle(
            (dev + 0.15, start),
            0.7, duration,
            linewidth=1.2,
            edgecolor="black",
            facecolor=color,
            alpha=0.85
        )
        ax.add_patch(rect)

        # 任务号
        ax.text(
            dev + 0.5,
            start + duration / 2,
            f"T{task+1}",
            ha="center",
            va="center",
            fontsize=12,
            color="black"
        )

    # --- 时间轴设置 ---
    ax.set_ylabel("Time", fontsize=12)
    ax.set_xlabel("Devices", fontsize=12)
    ax.set_xlim(0, num_devices)
    ax.set_ylim(0, max((r["start"]+r["predicted"]) for r in task_records))
    ax.invert_yaxis()  # 时间向下

    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()

    # 高清保存
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"甘特图(预测)已保存到: {save_path}")

    plt.show()