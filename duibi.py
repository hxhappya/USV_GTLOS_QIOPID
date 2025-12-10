import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import seaborn as sns
from scipy import signal

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ThreeAlgorithmAnalyzer:
    """三个算法数据分析和对比类"""

    def __init__(self):
        self.los_data = None
        self.gtlos_data = None
        self.gtlos_qiopid_data = None
        self.comparison_results = {}

    def load_data(self, los_filepath, gtlos_filepath, gtlos_qiopid_filepath):
        """加载三个算法保存的数据"""
        try:
            with open(los_filepath, 'rb') as f:
                self.los_data = pickle.load(f)
            print(f"成功加载LOS数据: {los_filepath}")

            with open(gtlos_filepath, 'rb') as f:
                self.gtlos_data = pickle.load(f)
            print(f"成功加载GTLOS数据: {gtlos_filepath}")

            with open(gtlos_qiopid_filepath, 'rb') as f:
                self.gtlos_qiopid_data = pickle.load(f)
            print(f"成功加载GTLOS-QIOPID数据: {gtlos_qiopid_filepath}")

            return True
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return False

    def _smooth_data(self, data, window_size=51):
        """使用Savitzky-Golay滤波器平滑数据"""
        if len(data) < window_size:
            return data
        return signal.savgol_filter(data, window_size, 3)

    def plot_trajectory_comparison(self):
        """绘制三个算法的轨迹对比图"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        # 绘制参考路径
        ax.plot(self.los_data['path_points'][:, 0], self.los_data['path_points'][:, 1],
                'g--', label='Reference Path', linewidth=1, alpha=0.8)

        # 绘制LOS轨迹
        los_traj = self.los_data['position']
        ax.plot([p[0] for p in los_traj], [p[1] for p in los_traj],
                'b-', label='LOS', linewidth=1, alpha=0.8)

        # 绘制GTLOS轨迹
        gtlos_traj = self.gtlos_data['position']
        ax.plot([p[0] for p in gtlos_traj], [p[1] for p in gtlos_traj],
                'orange', label='GTLOS', linewidth=1, alpha=0.8)

        # 绘制GTLOS-QIOPID轨迹
        gtlos_qiopid_traj = self.gtlos_qiopid_data['position']
        ax.plot([p[0] for p in gtlos_qiopid_traj], [p[1] for p in gtlos_qiopid_traj],
                'r-', label='GTLOS-QIOPID', linewidth=1, alpha=0.8)

        # 标记起点和终点
        ax.scatter(self.los_data['path_points'][0, 0], self.los_data['path_points'][0, 1],
                   c='green', marker='o', s=150, label='start', zorder=5, edgecolors='black')
        ax.scatter(self.los_data['path_points'][-1, 0], self.los_data['path_points'][-1, 1],
                   c='red', marker='X', s=150, label='end', zorder=5, edgecolors='black')

        # 添加图例和标签
        ax.set_xlabel('X (m)', fontsize=20)
        ax.set_ylabel('Y (m)', fontsize=20)
        # ax.set_title('三个算法轨迹对比', fontsize=20, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(-10, 100)
        ax.set_ylim(-10, 100)
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
         # 美化图例
        legend = ax.legend(fontsize=20, loc='upper right',
                          frameon=True, fancybox=True,
                          shadow=True, framealpha=0.95,
                          edgecolor='black')
        legend.get_frame().set_facecolor('white')
        plt.tight_layout()
        plt.savefig('three_algorithm_trajectory_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def plot_path_error_comparison(self):
        """绘制三个算法的路径偏差对比图"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        # 确保时间向量长度一致
        min_len = min(len(self.los_data['time']), len(self.gtlos_data['time']), len(self.gtlos_qiopid_data['time']))
        time_data = self.los_data['time'][:min_len]

        # 获取路径误差数据
        los_error = self.los_data['path_error'][:min_len]
        gtlos_error = self.gtlos_data['path_error'][:min_len]
        gtlos_qiopid_error = self.gtlos_qiopid_data['path_error'][:min_len]

        # 平滑数据
        los_error_smooth = self._smooth_data(los_error)
        gtlos_error_smooth = self._smooth_data(gtlos_error)
        gtlos_qiopid_error_smooth = self._smooth_data(gtlos_qiopid_error)

        # 绘制三个算法的路径偏差
        ax.plot(time_data, los_error_smooth, 'b-', label='LOS', linewidth=1.5, alpha=0.8)
        ax.plot(time_data, gtlos_error_smooth, 'orange', label='GTLOS', linewidth=1.5, alpha=0.8)
        ax.plot(time_data, gtlos_qiopid_error_smooth, 'r-', label='GTLOS-QIOPID', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('时间 (s)', fontsize=14)
        ax.set_ylabel('路径偏差 (m)', fontsize=14)
        ax.set_title('三个算法路径偏差对比', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        los_mean = np.mean(los_error)
        gtlos_mean = np.mean(gtlos_error)
        gtlos_qiopid_mean = np.mean(gtlos_qiopid_error)

        improvement_gtlos = (los_mean - gtlos_mean) / los_mean * 100 if los_mean != 0 else 0
        improvement_gtlos_qiopid = (los_mean - gtlos_qiopid_mean) / los_mean * 100 if los_mean != 0 else 0

        textstr = '\n'.join((
            f'LOS 平均偏差: {los_mean:.3f}m',
            f'GTLOS 平均偏差: {gtlos_mean:.3f}m (改进: {improvement_gtlos:.1f}%)',
            f'GTLOS-QIOPID 平均偏差: {gtlos_qiopid_mean:.3f}m (改进: {improvement_gtlos_qiopid:.1f}%)'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig('three_algorithm_path_error_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def plot_heading_error_comparison(self):
        """绘制三个算法的航向误差对比图"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        # 确保时间向量长度一致
        min_len = min(len(self.los_data['time']), len(self.gtlos_data['time']), len(self.gtlos_qiopid_data['time']))
        time_data = self.los_data['time'][:min_len]

        # 获取航向误差数据
        los_heading_error = np.abs(self.los_data['heading_error'][:min_len])
        gtlos_heading_error = np.abs(self.gtlos_data['heading_error'][:min_len])
        gtlos_qiopid_heading_error = np.abs(self.gtlos_qiopid_data['heading_error'][:min_len])

        # 平滑数据
        los_heading_smooth = self._smooth_data(los_heading_error)
        gtlos_heading_smooth = self._smooth_data(gtlos_heading_error)
        gtlos_qiopid_heading_smooth = self._smooth_data(gtlos_qiopid_heading_error)

        # 绘制三个算法的航向误差
        ax.plot(time_data, los_heading_smooth, 'b-', label='LOS', linewidth=2.5, alpha=0.8)
        ax.plot(time_data, gtlos_heading_smooth, 'orange', label='GTLOS', linewidth=2.5, alpha=0.8)
        ax.plot(time_data, gtlos_qiopid_heading_smooth, 'r-', label='GTLOS-QIOPID', linewidth=2.5, alpha=0.8)

        ax.set_xlabel('时间 (s)', fontsize=14)
        ax.set_ylabel('航向误差绝对值 (rad)', fontsize=14)
        ax.set_title('三个算法航向误差对比', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        los_mean = np.mean(los_heading_error)
        gtlos_mean = np.mean(gtlos_heading_error)
        gtlos_qiopid_mean = np.mean(gtlos_qiopid_heading_error)

        improvement_gtlos = (los_mean - gtlos_mean) / los_mean * 100 if los_mean != 0 else 0
        improvement_gtlos_qiopid = (los_mean - gtlos_qiopid_mean) / los_mean * 100 if los_mean != 0 else 0

        textstr = '\n'.join((
            f'LOS 平均误差: {los_mean:.3f}rad',
            f'GTLOS 平均误差: {gtlos_mean:.3f}rad (改进: {improvement_gtlos:.1f}%)',
            f'GTLOS-QIOPID 平均误差: {gtlos_qiopid_mean:.3f}rad (改进: {improvement_gtlos_qiopid:.1f}%)'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig('three_algorithm_heading_error_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def plot_ground_speed_comparison(self):
        """绘制三个算法的对地速度对比图"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        # 确保时间向量长度一致
        min_len = min(len(self.los_data['time']), len(self.gtlos_data['time']), len(self.gtlos_qiopid_data['time']))
        time_data = self.los_data['time'][:min_len]

        # 计算对地速度
        los_speed = np.sqrt(self.los_data['velocity'][:min_len, 0]**2 + self.los_data['velocity'][:min_len, 1]**2)
        gtlos_speed = np.sqrt(self.gtlos_data['velocity'][:min_len, 0]**2 + self.gtlos_data['velocity'][:min_len, 1]**2)
        gtlos_qiopid_speed = np.sqrt(self.gtlos_qiopid_data['velocity'][:min_len, 0]**2 + self.gtlos_qiopid_data['velocity'][:min_len, 1]**2)

        # 平滑数据
        los_speed_smooth = self._smooth_data(los_speed, window_size=101)
        gtlos_speed_smooth = self._smooth_data(gtlos_speed, window_size=101)
        gtlos_qiopid_speed_smooth = self._smooth_data(gtlos_qiopid_speed, window_size=101)

        # 绘制三个算法的对地速度
        ax.plot(time_data, los_speed_smooth, 'b-', label='LOS', linewidth=2.5, alpha=0.8)
        ax.plot(time_data, gtlos_speed_smooth, 'orange', label='GTLOS', linewidth=2.5, alpha=0.8)
        ax.plot(time_data, gtlos_qiopid_speed_smooth, 'r-', label='GTLOS-QIOPID', linewidth=2.5, alpha=0.8)

        ax.set_xlabel('时间 (s)', fontsize=14)
        ax.set_ylabel('对地速度 (m/s)', fontsize=14)
        ax.set_title('三个算法对地速度对比', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        los_std = np.std(los_speed)
        gtlos_std = np.std(gtlos_speed)
        gtlos_qiopid_std = np.std(gtlos_qiopid_speed)

        improvement_gtlos = (los_std - gtlos_std) / los_std * 100 if los_std != 0 else 0
        improvement_gtlos_qiopid = (los_std - gtlos_qiopid_std) / los_std * 100 if los_std != 0 else 0

        textstr = '\n'.join((
            f'LOS 速度标准差: {los_std:.3f}m/s',
            f'GTLOS 速度标准差: {gtlos_std:.3f}m/s (改进: {improvement_gtlos:.1f}%)',
            f'GTLOS-QIOPID 速度标准差: {gtlos_qiopid_std:.3f}m/s (改进: {improvement_gtlos_qiopid:.1f}%)'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig('three_algorithm_ground_speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def plot_thrust_torque_comparison(self):
        """绘制三个算法的推进力矩对比图"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        # 确保时间向量长度一致
        min_len = min(len(self.los_data['time']), len(self.gtlos_data['time']), len(self.gtlos_qiopid_data['time']))
        time_data = self.los_data['time'][:min_len]

        # 计算推进力矩（n1 - n2）
        los_torque = self.los_data['control_input'][:min_len, 0] - self.los_data['control_input'][:min_len, 1]
        gtlos_torque = self.gtlos_data['control_input'][:min_len, 0] - self.gtlos_data['control_input'][:min_len, 1]
        gtlos_qiopid_torque = self.gtlos_qiopid_data['control_input'][:min_len, 0] - self.gtlos_qiopid_data['control_input'][:min_len, 1]

        # 平滑数据
        los_torque_smooth = self._smooth_data(los_torque, window_size=151)
        gtlos_torque_smooth = self._smooth_data(gtlos_torque, window_size=151)
        gtlos_qiopid_torque_smooth = self._smooth_data(gtlos_qiopid_torque, window_size=151)

        # 绘制三个算法的推进力矩
        ax.plot(time_data, los_torque_smooth, 'b-', label='LOS', linewidth=2.5, alpha=0.8)
        ax.plot(time_data, gtlos_torque_smooth, 'orange', label='GTLOS', linewidth=2.5, alpha=0.8)
        ax.plot(time_data, gtlos_qiopid_torque_smooth, 'r-', label='GTLOS-QIOPID', linewidth=2.5, alpha=0.8)

        ax.set_xlabel('时间 (s)', fontsize=14)
        ax.set_ylabel('推进力矩 (N·m)', fontsize=14)
        ax.set_title('三个算法推进力矩对比', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 添加统计信息
        los_std = np.std(los_torque)
        gtlos_std = np.std(gtlos_torque)
        gtlos_qiopid_std = np.std(gtlos_qiopid_torque)

        improvement_gtlos = (los_std - gtlos_std) / los_std * 100 if los_std != 0 else 0
        improvement_gtlos_qiopid = (los_std - gtlos_qiopid_std) / los_std * 100 if los_std != 0 else 0

        textstr = '\n'.join((
            f'LOS 力矩标准差: {los_std:.3f}N·m',
            f'GTLOS 力矩标准差: {gtlos_std:.3f}N·m (改进: {improvement_gtlos:.1f}%)',
            f'GTLOS-QIOPID 力矩标准差: {gtlos_qiopid_std:.3f}N·m (改进: {improvement_gtlos_qiopid:.1f}%)'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig('three_algorithm_thrust_torque_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def plot_performance_metrics_comparison(self):
        """绘制三个算法的性能指标对比图"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 获取性能指标
        los_metrics = self.los_data['performance_metrics']
        gtlos_metrics = self.gtlos_data['performance_metrics']
        gtlos_qiopid_metrics = self.gtlos_qiopid_data['performance_metrics']

        # 1. 路径跟踪性能指标
        metrics_names = ['RMSE', 'MAE', 'Max_AE']
        los_values = [los_metrics['RMSE'], los_metrics['MAE'], los_metrics['Max_AE']]
        gtlos_values = [gtlos_metrics['RMSE'], gtlos_metrics['MAE'], gtlos_metrics['Max_AE']]
        gtlos_qiopid_values = [gtlos_qiopid_metrics['RMSE'], gtlos_qiopid_metrics['MAE'], gtlos_qiopid_metrics['Max_AE']]

        x = np.arange(len(metrics_names))
        width = 0.25

        axes[0, 0].bar(x - width, los_values, width, label='LOS', alpha=0.8)
        axes[0, 0].bar(x, gtlos_values, width, label='GTLOS', alpha=0.8)
        axes[0, 0].bar(x + width, gtlos_qiopid_values, width, label='GTLOS-QIOPID', alpha=0.8)
        axes[0, 0].set_xlabel('性能指标', fontsize=12)
        axes[0, 0].set_ylabel('误差值 (m)', fontsize=12)
        axes[0, 0].set_title('路径跟踪性能指标对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_names)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # 2. 控制性能指标
        control_metrics = ['C1', 'C2', 'J']
        los_control = [los_metrics['C1'], los_metrics['C2'], los_metrics['J']]
        gtlos_control = [gtlos_metrics['C1'], gtlos_metrics['C2'], gtlos_metrics['J']]
        gtlos_qiopid_control = [gtlos_qiopid_metrics['C1'], gtlos_qiopid_metrics['C2'], gtlos_qiopid_metrics['J']]

        axes[0, 1].bar(x - width, los_control, width, label='LOS', alpha=0.8)
        axes[0, 1].bar(x, gtlos_control, width, label='GTLOS', alpha=0.8)
        axes[0, 1].bar(x + width, gtlos_qiopid_control, width, label='GTLOS-QIOPID', alpha=0.8)
        axes[0, 1].set_xlabel('性能指标', fontsize=12)
        axes[0, 1].set_ylabel('性能值', fontsize=12)
        axes[0, 1].set_title('控制性能指标对比', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(control_metrics)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. 改进百分比雷达图
        axes[1, 0].axis('off')

        # 计算改进百分比（相对于LOS）
        improvement_gtlos = [
            (los_metrics['RMSE'] - gtlos_metrics['RMSE']) / los_metrics['RMSE'] * 100,
            (los_metrics['MAE'] - gtlos_metrics['MAE']) / los_metrics['MAE'] * 100,
            (los_metrics['C1'] - gtlos_metrics['C1']) / los_metrics['C1'] * 100
        ]

        improvement_gtlos_qiopid = [
            (los_metrics['RMSE'] - gtlos_qiopid_metrics['RMSE']) / los_metrics['RMSE'] * 100,
            (los_metrics['MAE'] - gtlos_qiopid_metrics['MAE']) / los_metrics['MAE'] * 100,
            (los_metrics['C1'] - gtlos_qiopid_metrics['C1']) / los_metrics['C1'] * 100
        ]

        categories = ['RMSE改进', 'MAE改进', 'C1改进']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        improvement_gtlos += improvement_gtlos[:1]
        improvement_gtlos_qiopid += improvement_gtlos_qiopid[:1]

        ax_radar = fig.add_subplot(2, 2, 4, polar=True)
        ax_radar.plot(angles, improvement_gtlos, 'o-', linewidth=2, label='GTLOS', alpha=0.8)
        ax_radar.fill(angles, improvement_gtlos, alpha=0.25)
        ax_radar.plot(angles, improvement_gtlos_qiopid, 'o-', linewidth=2, label='GTLOS-QIOPID', alpha=0.8)
        ax_radar.fill(angles, improvement_gtlos_qiopid, alpha=0.25)
        ax_radar.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax_radar.set_title('相对于LOS的改进百分比', fontsize=14, fontweight='bold')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.tight_layout()
        plt.savefig('three_algorithm_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def generate_comprehensive_report(self):
        """生成综合对比报告"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        los_metrics = self.los_data['performance_metrics']
        gtlos_metrics = self.gtlos_data['performance_metrics']
        gtlos_qiopid_metrics = self.gtlos_qiopid_data['performance_metrics']

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""
        三个算法综合对比分析报告
        生成时间: {timestamp}
        ==============================================

        性能指标对比:

        1. 路径跟踪性能:
           RMSE: LOS={los_metrics['RMSE']:.4f}m, GTLOS={gtlos_metrics['RMSE']:.4f}m, GTLOS-QIOPID={gtlos_qiopid_metrics['RMSE']:.4f}m
           MAE: LOS={los_metrics['MAE']:.4f}m, GTLOS={gtlos_metrics['MAE']:.4f}m, GTLOS-QIOPID={gtlos_qiopid_metrics['MAE']:.4f}m
           Max AE: LOS={los_metrics['Max_AE']:.4f}m, GTLOS={gtlos_metrics['Max_AE']:.4f}m, GTLOS-QIOPID={gtlos_qiopid_metrics['Max_AE']:.4f}m

        2. 控制性能:
           C1: LOS={los_metrics['C1']:.2f}, GTLOS={gtlos_metrics['C1']:.2f}, GTLOS-QIOPID={gtlos_qiopid_metrics['C1']:.2f}
           C2: LOS={los_metrics['C2']:.2f}, GTLOS={gtlos_metrics['C2']:.2f}, GTLOS-QIOPID={gtlos_qiopid_metrics['C2']:.2f}
           J: LOS={los_metrics['J']:.2f}, GTLOS={gtlos_metrics['J']:.2f}, GTLOS-QIOPID={gtlos_qiopid_metrics['J']:.2f}

        3. 改进百分比 (相对于LOS):
           GTLOS RMSE改进: {(los_metrics['RMSE']-gtlos_metrics['RMSE'])/los_metrics['RMSE']*100:.1f}%
           GTLOS-QIOPID RMSE改进: {(los_metrics['RMSE']-gtlos_qiopid_metrics['RMSE'])/los_metrics['RMSE']*100:.1f}%
           GTLOS C1改进: {(los_metrics['C1']-gtlos_metrics['C1'])/los_metrics['C1']*100:.1f}%
           GTLOS-QIOPID C1改进: {(los_metrics['C1']-gtlos_qiopid_metrics['C1'])/los_metrics['C1']*100:.1f}%
        """

        # 保存报告到文件
        filename = f"three_algorithm_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"综合报告已保存到 {filename}")
        return report

    def run_complete_analysis(self, los_filepath, gtlos_filepath, gtlos_qiopid_filepath):
        """运行完整的分析流程"""
        print("开始三个算法对比分析...")

        # 加载数据
        if not self.load_data(los_filepath, gtlos_filepath, gtlos_qiopid_filepath):
            return

        # 绘制各个对比图
        print("\n1. 绘制轨迹对比图...")
        self.plot_trajectory_comparison()

        print("\n2. 绘制路径偏差对比图...")
        self.plot_path_error_comparison()

        print("\n3. 绘制航向误差对比图...")
        self.plot_heading_error_comparison()

        print("\n4. 绘制对地速度对比图...")
        self.plot_ground_speed_comparison()

        print("\n5. 绘制推进力矩对比图...")
        self.plot_thrust_torque_comparison()

        print("\n6. 绘制性能指标对比图...")
        self.plot_performance_metrics_comparison()

        print("\n7. 生成综合报告...")
        self.generate_comprehensive_report()

        print("\n分析完成！所有图表和报告已保存到当前目录。")

# 主程序
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = ThreeAlgorithmAnalyzer()

    # 指定三个算法的数据文件路径
    los_data_file = "data\los_simulation_data_20250922_204913.pkl"  # LOS算法数据
    gtlos_data_file = "data\gtlos_simulation_data_20250922_204955.pkl"  # GTLOS算法数据
    gtlos_qiopid_data_file = "data\gtlos-qiopid_simulation_data_20250922_205007.pkl"  # GTLOS-QIOPID算法数据

    # 运行分析
    analyzer.run_complete_analysis(los_data_file, gtlos_data_file, gtlos_qiopid_data_file)
