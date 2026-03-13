import numpy as np
import matplotlib.pyplot as plt
import copy
from evo.core import trajectory, sync, metrics
from evo.tools import file_interface

# ================= 数据准备与对齐 =================
# 读取你刚跑出来的真值和估算轨迹
gt = file_interface.read_tum_trajectory_file("/workspace/groundtruth.txt")
vo = file_interface.read_tum_trajectory_file("/workspace/KeyFrameTrajectory.txt")

# 时间戳同步匹配 (匹配两者的点)
max_diff = 0.05
traj_ref, traj_est = sync.associate_trajectories(gt, vo, max_diff)

# 【核心步骤 1：处理左上图的坐标偏移】
# 真值的原始坐标在千万级别(10^7)，为了达到老师图表中 [-300, 100] 的合理范围，将起点重置为 0
offset_x, offset_y = traj_ref.positions_xyz[0,0], traj_ref.positions_xyz[0,1]
gt_x = traj_ref.positions_xyz[:,0] - offset_x
gt_y = traj_ref.positions_xyz[:,1] - offset_y

# 未对齐的 VO 轨迹也把起点拉到一起
vo_x_un = traj_est.positions_xyz[:,0] - traj_est.positions_xyz[0,0]
vo_y_un = traj_est.positions_xyz[:,1] - traj_est.positions_xyz[0,1]

# 【核心步骤 2：处理右上图的 Sim(3) 缩放与对齐】
traj_est_aligned = copy.deepcopy(traj_est)
traj_est_aligned.align(traj_ref, correct_scale=True)
# 对齐后的 VO 坐标同样减去真值的偏移量，以保证画面重合
vo_x_al = traj_est_aligned.positions_xyz[:,0] - offset_x
vo_y_al = traj_est_aligned.positions_xyz[:,1] - offset_y

# 【核心步骤 3：计算下面两张图需要的 ATE 误差数据】
data = (traj_ref, traj_est_aligned)
ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
ape_metric.process_data(data)
ape_stats = ape_metric.get_all_statistics()
ape_error = ape_metric.error

# ================= 开始精细化绘图 (完全还原案例) =================
fig, axs = plt.subplots(2, 2, figsize=(14, 14), dpi=150)

# 1. 【左上图】：未对齐轨迹 (Before Alignment)
axs[0,0].plot(gt_x, gt_y, color='green', linewidth=2, label='Ground Truth')
axs[0,0].plot(vo_x_un, vo_y_un, color='red', linestyle='--', linewidth=2, label='VO (Unaligned)')
axs[0,0].set_title('2D Trajectory - Before Alignment ( HKisland_GNSS03)', fontsize=14)
axs[0,0].set_xlabel('X [m]', fontsize=12)
axs[0,0].set_ylabel('Y [m]', fontsize=12)
axs[0,0].legend(loc='upper left')
axs[0,0].grid(True, alpha=0.3)

# 2. 【右上图】：Sim(3) 对齐后轨迹 (After Alignment)
axs[0,1].plot(gt_x, gt_y, color='green', linewidth=2, label='Ground Truth')
axs[0,1].plot(vo_x_al, vo_y_al, color='blue', linewidth=2, label='VO (Aligned)')
axs[0,1].set_title('2D Trajectory - After Sim(3) Alignment ( HKisland_GNSS03)', fontsize=14)
axs[0,1].set_xlabel('X [m]', fontsize=12)
axs[0,1].set_ylabel('Y [m]', fontsize=12)
axs[0,1].legend(loc='upper left')
axs[0,1].grid(True, alpha=0.3)

# 3. 【左下图】：ATE 误差直方图 (Error Distribution)
axs[1,0].hist(ape_error, bins=35, color='#7b9cb9', edgecolor='black')
axs[1,0].axvline(ape_stats['mean'], color='red', linestyle='--', linewidth=2.5, label=f"Mean: {ape_stats['mean']:.2f} m")
axs[1,0].axvline(ape_stats['median'], color='orange', linestyle='--', linewidth=2.5, label=f"Median: {ape_stats['median']:.2f} m")
axs[1,0].set_title('Absolute Trajectory Error Distribution', fontsize=14)
axs[1,0].set_xlabel('ATE [m]', fontsize=12)
axs[1,0].set_ylabel('Frequency', fontsize=12)
axs[1,0].legend(loc='upper right')
axs[1,0].grid(True, alpha=0.3)

# 4. 【右下图】：随索引变化的 ATE 误差曲线 (Error Along Trajectory)
axs[1,1].plot(ape_error, color='blue', linewidth=1.2)
axs[1,1].fill_between(range(len(ape_error)), ape_error, color='#d2e5e5', alpha=0.6)
axs[1,1].set_title('ATE Error Along Trajectory', fontsize=14)
axs[1,1].set_xlabel('Matched Pose Index', fontsize=12)
axs[1,1].set_ylabel('ATE [m]', fontsize=12)
axs[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/Final_Report_Plot.png')
print("✅  /workspace/Final_Report_Plot.png")