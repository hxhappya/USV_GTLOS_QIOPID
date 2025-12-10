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

    def _enhance_smooth_data(self, data, window_size=151, polyorder=3):
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

    def _calculate_improvements(self):
        """计算改进百分比"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            return 0, 0

        min_len = min(len(self.los_data['time']),
                     len(self.gtlos_data['time']),
                     len(self.gtlos_qiopid_data['time']))

        los_error = self.los_data['path_error'][:min_len]
        gtlos_error = self.gtlos_data['path_error'][:min_len]
        gtlos_qiopid_error = self.gtlos_qiopid_data['path_error'][:min_len]

        los_mean = np.mean(los_error)
        gtlos_mean = np.mean(gtlos_error)
        gtlos_qiopid_mean = np.mean(gtlos_qiopid_error)

        improvement_gtlos = (los_mean - gtlos_mean) / los_mean * 100 if los_mean != 0 else 0
        improvement_gtlos_qiopid = (los_mean - gtlos_qiopid_mean) / los_mean * 100 if los_mean != 0 else 0

        return improvement_gtlos, improvement_gtlos_qiopid

    def plot_path_error_comparison(self):
        """绘制美化后的三个算法路径偏差对比图"""
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
        los_error = self.los_data['path_error'][:min_len]
        gtlos_error = self.gtlos_data['path_error'][:min_len]
        gtlos_qiopid_error = self.gtlos_qiopid_data['path_error'][:min_len]

        # 重新采样数据以减少波动
        if len(time_data) > 100:
            time_smooth, los_smooth = self._resample_data(time_data, los_error, 300)
            _, gtlos_smooth = self._resample_data(time_data, gtlos_error, 300)
            _, gtlos_qiopid_smooth = self._resample_data(time_data, gtlos_qiopid_error, 300)
        else:
            time_smooth, los_smooth = time_data, los_error
            gtlos_smooth, gtlos_qiopid_smooth = gtlos_error, gtlos_qiopid_error

        # 增强平滑处理
        los_enhanced = self._enhance_smooth_data(los_smooth, window_size=101)
        gtlos_enhanced = self._enhance_smooth_data(gtlos_smooth, window_size=101)
        gtlos_qiopid_enhanced = self._enhance_smooth_data(gtlos_qiopid_smooth, window_size=101)

        # 计算统计数据（使用原始数据）
        los_mean = np.mean(los_error)
        gtlos_mean = np.mean(gtlos_error)
        gtlos_qiopid_mean = np.mean(gtlos_qiopid_error)

        self.improvement_gtlos, self.improvement_gtlos_qiopid = self._calculate_improvements()

        # 绘制平滑后的曲线 - 使用更专业的颜色和样式
        line_los = ax.plot(time_smooth, los_enhanced, color='blue', linewidth=1.5,
                          label='LOS', alpha=0.9, marker='', linestyle='-')
        line_gtlos = ax.plot(time_smooth, gtlos_enhanced, color='red', linewidth=1.5,
                           label='GTLOS', alpha=0.9, marker='', linestyle='-')
        line_gtlos_qiopid = ax.plot(time_smooth, gtlos_qiopid_enhanced, color='green', linewidth=1.5,
                                  label='GTLS-010PID', alpha=0.95, marker='', linestyle='-')

        # 添加填充区域显示波动范围（轻微显示）
        # ax.fill_between(time_smooth, los_enhanced * 0.95, los_enhanced * 1.05,
        #                alpha=0.2, color='#2E86AB')
        # ax.fill_between(time_smooth, gtlos_enhanced * 0.97, gtlos_enhanced * 1.03,
        #                alpha=0.2, color='#A23B72')
        # ax.fill_between(time_smooth, gtlos_qiopid_enhanced * 0.98, gtlos_qiopid_enhanced * 1.02,
        #                alpha=0.2, color='#18A999')

        # 设置坐标轴标签和标题
        ax.set_xlabel('time (s)', fontsize=20, fontweight='bold', labelpad=10)
        ax.set_ylabel('deviation (m)', fontsize=20, fontweight='bold', labelpad=10)
        # ax.set_title('三个算法路径偏差对比', fontsize=20, fontweight='bold', pad=20)

        # 优化坐标轴范围
        y_max = max(np.max(los_enhanced), np.max(gtlos_enhanced), np.max(gtlos_qiopid_enhanced))
        ax.set_ylim(0, y_max * 1.15)  # 留出空间显示图例和统计信息

        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # 美化图例
        legend = ax.legend(fontsize=20, loc='upper right',
                          frameon=True, fancybox=True,
                          shadow=True, framealpha=0.95,
                          edgecolor='black')
        legend.get_frame().set_facecolor('white')

        # # 添加改进效果的标注（更突出显示）
        # textstr = '\n'.join((
        #     f'LOS 平均偏差: {los_mean:.3f}m',
        #     f'GTLS 平均偏差: {gtlos_mean:.3f}m (改进: {self.improvement_gtlos:.1f}%)',
        #     f'GTLS-010PID 平均偏差: {gtlos_qiopid_mean:.3f}m (改进: {self.improvement_gtlos_qiopid:.1f}%)'
        # ))

        # # 使用更醒目的文本框
        # props = dict(boxstyle='round', facecolor='gold', alpha=0.9,
        #             edgecolor='orange', linewidth=2)
        # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=13,
        #         verticalalignment='top', bbox=props, fontweight='bold')

        # 添加改进箭头标注
        if self.improvement_gtlos_qiopid > 40:  # 如果改进显著，添加箭头标注
            ax.annotate('Significant improvement', xy=(time_smooth[len(time_smooth)//2],
                                     gtlos_qiopid_enhanced[len(time_smooth)//2]),
                       xytext=(time_smooth[len(time_smooth)//2],
                             gtlos_qiopid_enhanced[len(time_smooth)//2] + y_max * 0.2),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=12, fontweight='bold', color='red',
                       ha='center')

        # 优化布局
        plt.tight_layout()

        # 保存高质量图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'enhanced_path_error_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')

        plt.show()

        print("路径偏差对比图已美化完成！")
        print(f"LOS平均偏差: {los_mean:.3f}m")
        print(f"GTLOS平均偏差: {gtlos_mean:.3f}m (改进: {self.improvement_gtlos:.1f}%)")
        print(f"GTLOS-QIOPID平均偏差: {gtlos_qiopid_mean:.3f}m (改进: {self.improvement_gtlos_qiopid:.1f}%)")

        return fig

    def plot_enhanced_comparison_summary(self):
        """绘制增强的综合对比总结图"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        # 确保改进百分比已经计算
        if self.improvement_gtlos == 0 and self.improvement_gtlos_qiopid == 0:
            self.improvement_gtlos, self.improvement_gtlos_qiopid = self._calculate_improvements()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('三个算法性能综合对比', fontsize=20, fontweight='bold', y=0.95)

        # 1. 路径偏差对比（主图）
        min_len = min(len(self.los_data['time']),
                     len(self.gtlos_data['time']),
                     len(self.gtlos_qiopid_data['time']))
        time_data = self.los_data['time'][:min_len]

        # 平滑处理
        time_smooth, los_smooth = self._resample_data(time_data, self.los_data['path_error'][:min_len])
        _, gtlos_smooth = self._resample_data(time_data, self.gtlos_data['path_error'][:min_len])
        _, gtlos_qiopid_smooth = self._resample_data(time_data, self.gtlos_qiopid_data['path_error'][:min_len])

        ax1.plot(time_smooth, self._enhance_smooth_data(los_smooth),
                label='LOS', linewidth=2, color='#2E86AB')
        ax1.plot(time_smooth, self._enhance_smooth_data(gtlos_smooth),
                label='GTLOS', linewidth=2, color='#A23B72')
        ax1.plot(time_smooth, self._enhance_smooth_data(gtlos_qiopid_smooth),
                label='GTLS-010PID', linewidth=3, color='#18A999')
        ax1.set_title('路径偏差对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('路径偏差 (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 平均偏差柱状图
        los_mean = np.mean(self.los_data['path_error'])
        gtlos_mean = np.mean(self.gtlos_data['path_error'])
        gtlos_qiopid_mean = np.mean(self.gtlos_qiopid_data['path_error'])

        means = [los_mean, gtlos_mean, gtlos_qiopid_mean]
        algorithms = ['LOS', 'GTLOS', 'GTLS-010PID']
        colors = ['#2E86AB', '#A23B72', '#18A999']

        bars = ax2.bar(algorithms, means, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('平均路径偏差对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('平均偏差 (m)')

        # 在柱子上添加数值
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}m', ha='center', va='bottom', fontweight='bold')

        # 3. 改进百分比饼图
        improvements = [0, self.improvement_gtlos, self.improvement_gtlos_qiopid]
        # 只显示正改进的算法
        positive_improvements = []
        positive_labels = []
        positive_colors = []

        for i, (imp, label, color) in enumerate(zip(improvements[1:], algorithms[1:], colors[1:])):
            if imp > 0:
                positive_improvements.append(imp)
                positive_labels.append(label)
                positive_colors.append(color)

        if positive_improvements:
            wedges, texts, autotexts = ax3.pie(positive_improvements, labels=positive_labels,
                                              autopct='%1.1f%%', startangle=90,
                                              colors=positive_colors)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax3.text(0.5, 0.5, '无显著改进', ha='center', va='center', fontsize=14)

        ax3.set_title('相对于LOS的改进百分比', fontsize=14, fontweight='bold')

        # 4. 稳定性对比（标准差）
        stds = [np.std(self.los_data['path_error']),
               np.std(self.gtlos_data['path_error']),
               np.std(self.gtlos_qiopid_data['path_error'])]

        bars_std = ax4.bar(algorithms, stds, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_title('路径偏差稳定性对比', fontsize=14, fontweight='bold')
        ax4.set_ylabel('标准差')

        # 在柱子上添加数值
        for bar, std in zip(bars_std, stds):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{std:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'enhanced_comprehensive_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

        return fig

# 主程序
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = ThreeAlgorithmAnalyzer()

    # 指定三个算法的数据文件路径
    los_data_file = "data\los_circle_simulation_data_20251210_110140.pkl"  # LOS算法数据
    gtlos_data_file = "data\gtlos_circle_simulation_data_20250923_194827.pkl"  # GTLOS算法数据
    gtlos_qiopid_data_file = "data\gtlos-qiopid_circle_simulation_data_20250923_194626.pkl"  # GTLOS-QIOPID算法数据
    # 加载数据
    if analyzer.load_data(los_data_file, gtlos_data_file, gtlos_qiopid_data_file):
        # 绘制美化后的路径偏差对比图
        analyzer.plot_path_error_comparison()

        # 绘制综合对比图
        analyzer.plot_enhanced_comparison_summary()

        print("所有美化图表已生成完成！")
    else:
        print("数据加载失败，请检查文件路径是否正确。")
