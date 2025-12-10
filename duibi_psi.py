import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import seaborn as sns
from scipy import signal
from scipy.interpolate import interp1d

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
        self.improvement_gtlos = 0
        self.improvement_gtlos_qiopid = 0

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

    def _enhance_smooth_data(self, data, window_size=10, polyorder=2):
        """增强的数据平滑方法"""
        if len(data) < window_size:
            window_size = len(data) - 1 if len(data) % 2 == 0 else len(data)
            if window_size < 3:
                return data

        # 使用更大的窗口进行平滑
        smoothed = signal.savgol_filter(data, window_size, polyorder)

        # 额外的低通滤波
        b, a = signal.butter(2, 0.1)
        smoothed = signal.filtfilt(b, a, smoothed)

        return smoothed

    def _resample_data(self, time, data, num_points=500):
        """重新采样数据以减少波动"""
        if len(time) < 10:
            return time, data

        # 创建插值函数
        f = interp1d(time, data, kind='linear', fill_value='extrapolate')

        # 创建新的时间点
        new_time = np.linspace(time[0], time[-1], num_points)
        new_data = f(new_time)

        return new_time, new_data

    def _rad_to_deg(self, rad_data):
        """将弧度转换为度数"""
        return np.degrees(rad_data)

    def _calculate_heading_improvements(self):
        """计算航向误差的改进百分比（使用度数）"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            return 0, 0, 0, 0, 0

        min_len = min(len(self.los_data['time']),
                     len(self.gtlos_data['time']),
                     len(self.gtlos_qiopid_data['time']))

        # 获取航向误差数据（绝对值）并转换为度数
        los_heading_error = np.abs(self._rad_to_deg(self.los_data['heading_error'][:min_len]))
        gtlos_heading_error = np.abs(self._rad_to_deg(self.gtlos_data['heading_error'][:min_len]))
        gtlos_qiopid_heading_error = np.abs(self._rad_to_deg(self.gtlos_qiopid_data['heading_error'][:min_len]))

        los_mean = np.mean(los_heading_error)
        gtlos_mean = np.mean(gtlos_heading_error)
        gtlos_qiopid_mean = np.mean(gtlos_qiopid_heading_error)

        improvement_gtlos = (los_mean - gtlos_mean) / los_mean * 100 if los_mean != 0 else 0
        improvement_gtlos_qiopid = (los_mean - gtlos_qiopid_mean) / los_mean * 100 if los_mean != 0 else 0

        return improvement_gtlos, improvement_gtlos_qiopid, los_mean, gtlos_mean, gtlos_qiopid_mean

    def plot_heading_angle_comparison(self):
        """绘制三个算法的艏向角对比图"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        # 创建更专业的图形
        fig, ax = plt.subplots(figsize=(16, 10))

        # 设置背景色
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')

        # 确保时间向量长度一致
        min_len = min(len(self.los_data['time']),
                     len(self.gtlos_data['time']),
                     len(self.gtlos_qiopid_data['time']))

        time_data = self.los_data['time'][:min_len]

        # 获取艏向角数据并转换为度数
        # 假设数据中包含'heading'字段，如果没有则需要从其他数据计算
        try:
            los_heading = self._rad_to_deg(self.los_data['heading'][:min_len])
            gtlos_heading = self._rad_to_deg(self.gtlos_data['heading'][:min_len])
            gtlos_qiopid_heading = self._rad_to_deg(self.gtlos_qiopid_data['heading'][:min_len])
        except KeyError:
            # 如果没有heading字段，尝试从速度向量计算航向角
            print("警告：数据中没有'heading'字段，尝试从速度向量计算航向角")
            los_heading = self._rad_to_deg(np.arctan2(self.los_data['velocity'][:min_len, 1],
                                                     self.los_data['velocity'][:min_len, 0]))
            gtlos_heading = self._rad_to_deg(np.arctan2(self.gtlos_data['velocity'][:min_len, 1],
                                                       self.gtlos_data['velocity'][:min_len, 0]))
            gtlos_qiopid_heading = self._rad_to_deg(np.arctan2(self.gtlos_qiopid_data['velocity'][:min_len, 1],
                                                             self.gtlos_qiopid_data['velocity'][:min_len, 0]))

        # 重新采样数据以减少波动
        if len(time_data) > 100:
            time_smooth, los_smooth = self._resample_data(time_data, los_heading, 300)
            _, gtlos_smooth = self._resample_data(time_data, gtlos_heading, 300)
            _, gtlos_qiopid_smooth = self._resample_data(time_data, gtlos_qiopid_heading, 300)
        else:
            time_smooth, los_smooth = time_data, los_heading
            gtlos_smooth, gtlos_qiopid_smooth = gtlos_heading, gtlos_qiopid_heading

        # 增强平滑处理
        los_enhanced = self._enhance_smooth_data(los_smooth, window_size=101)
        gtlos_enhanced = self._enhance_smooth_data(gtlos_smooth, window_size=101)
        gtlos_qiopid_enhanced = self._enhance_smooth_data(gtlos_qiopid_smooth, window_size=101)

        # 绘制平滑后的曲线 - 使用更专业的颜色和样式
        ax.plot(time_smooth, los_enhanced, color='blue', linewidth=1.5,
                label='LOS', alpha=0.9, marker='', linestyle='-')
        ax.plot(time_smooth, gtlos_enhanced, color='red', linewidth=1.5,
                label='GTLOS', alpha=0.9, marker='', linestyle='-')
        ax.plot(time_smooth, gtlos_qiopid_enhanced, color='green', linewidth=1.5,
                label='GTLS-QIOPID', alpha=0.95, marker='', linestyle='-')

        # # 添加填充区域显示波动范围（轻微显示）
        # ax.fill_between(time_smooth, los_enhanced * 0.99, los_enhanced * 1.01,
        #                alpha=0.2, color='#2E86AB')
        # ax.fill_between(time_smooth, gtlos_enhanced * 0.995, gtlos_enhanced * 1.005,
        #                alpha=0.2, color='#A23B72')
        # ax.fill_between(time_smooth, gtlos_qiopid_enhanced * 0.998, gtlos_qiopid_enhanced * 1.002,
        #                alpha=0.2, color='#18A999')

        # 设置坐标轴标签和标题
        ax.set_xlabel('时间 (s)', fontsize=20, fontweight='bold', labelpad=10)
        ax.set_ylabel('艏向角 (°)', fontsize=20, fontweight='bold', labelpad=10)
        ax.set_title('三个算法艏向角对比', fontsize=20, fontweight='bold', pad=20)

        # 优化坐标轴范围
        y_min = min(np.min(los_enhanced), np.min(gtlos_enhanced), np.min(gtlos_qiopid_enhanced))
        y_max = max(np.max(los_enhanced), np.max(gtlos_enhanced), np.max(gtlos_qiopid_enhanced))
        ax.set_ylim(0, y_max * 1.05)

        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # 美化图例
        legend = ax.legend(fontsize=20, loc='upper right',
                          frameon=True, fancybox=True,
                          shadow=True, framealpha=0.95,
                          edgecolor='black')
        legend.get_frame().set_facecolor('white')

        # 添加统计信息
        los_mean = np.mean(los_heading)
        gtlos_mean = np.mean(gtlos_heading)
        gtlos_qiopid_mean = np.mean(gtlos_qiopid_heading)

        # textstr = '\n'.join((
        #     f'LOS 平均艏向角: {los_mean:.1f}°',
        #     f'GTLOS 平均艏向角: {gtlos_mean:.1f}°',
        #     f'GTLS-010PID 平均艏向角: {gtlos_qiopid_mean:.1f}°'
        # ))

        # # 使用更醒目的文本框
        # props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9,
        #             edgecolor='green', linewidth=2)
        # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=13,
        #         verticalalignment='top', bbox=props, fontweight='bold')

        # 优化布局
        plt.tight_layout()

        # 保存高质量图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'heading_angle_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')

        plt.show()

        print("艏向角对比图已生成完成！")
        print(f"LOS平均艏向角: {los_mean:.1f}°")
        print(f"GTLOS平均艏向角: {gtlos_mean:.1f}°")
        print(f"GTLOS-QIOPID平均艏向角: {gtlos_qiopid_mean:.1f}°")

        return fig

    def plot_enhanced_heading_summary(self):
        """绘制增强的航向误差综合对比总结图（使用度数）"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        fig, ax1 = plt.subplots( figsize=(16, 10))
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('#f8f9fa')

        # 1. 航向误差对比（主图）
        min_len = min(len(self.los_data['time']),
                     len(self.gtlos_data['time']),
                     len(self.gtlos_qiopid_data['time']))
        time_data = self.los_data['time'][:min_len]

        # 获取航向误差数据并转换为度数
        los_heading_error = (self._rad_to_deg(self.los_data['heading_error'][:min_len]))
        gtlos_heading_error = (self._rad_to_deg(self.gtlos_data['heading_error'][:min_len]))
        gtlos_qiopid_heading_error = (self._rad_to_deg(self.gtlos_qiopid_data['heading_error'][:min_len]))

        # 平滑处理
        time_smooth, los_smooth = self._resample_data(time_data, los_heading_error)
        _, gtlos_smooth = self._resample_data(time_data, gtlos_heading_error)
        _, gtlos_qiopid_smooth = self._resample_data(time_data, gtlos_qiopid_heading_error)

        ax1.plot(time_smooth, self._enhance_smooth_data(los_smooth),
                label='LOS', linewidth=1.5, color='blue', alpha=0.9, marker='', linestyle='-')
        ax1.plot(time_smooth, self._enhance_smooth_data(gtlos_smooth),
                label='GTLOS', linewidth=1.5, color='red', alpha=0.9, marker='', linestyle='-')
        ax1.plot(time_smooth, self._enhance_smooth_data(gtlos_qiopid_smooth),
                label='GTLS-QIOPID', linewidth=1.5, color='green', alpha=0.9, marker='', linestyle='-')
        # ax1.set_title('艏向误差对比', fontsize=20, fontweight='bold')
        ax1.set_xlabel('time (s)', fontsize=20, fontweight='bold', labelpad=10)
        ax1.set_ylabel('error (°)', fontsize=20, fontweight='bold', labelpad=10)
         # 设置网格
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.set_axisbelow(True)

        ax1.legend()


        # 美化图例
        legend = ax1.legend(fontsize=20, loc='upper right',
                          frameon=True, fancybox=True,
                          shadow=True, framealpha=0.95,
                          edgecolor='black')
        legend.get_frame().set_facecolor('white')
         # 优化布局
        plt.tight_layout()
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'enhanced_heading_comprehensive_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

        return fig

# 主程序
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = ThreeAlgorithmAnalyzer()

    # 指定三个算法的数据文件路径
    los_data_file = "data\los_simulation_data_20250922_204913.pkl"  # LOS算法数据
    gtlos_data_file = "data\gtlos_simulation_data_20250922_204955.pkl"  # GTLOS算法数据
    gtlos_qiopid_data_file = "data\gtlos-qiopid_simulation_data_20250922_205007.pkl"  # GTLOS-QIOPID算法数据

    # 加载数据
    if analyzer.load_data(los_data_file, gtlos_data_file, gtlos_qiopid_data_file):
        # 绘制艏向角对比图
        analyzer.plot_heading_angle_comparison()

        # 绘制航向误差综合对比图
        analyzer.plot_enhanced_heading_summary()

        print("所有航向相关图表已生成完成！")
    else:
        print("数据加载失败，请检查文件路径是否正确。")
