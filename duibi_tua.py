import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import seaborn as sns

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

    def _calculate_improvements(self):
        """计算改进百分比"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            return 0, 0

        min_len = min(len(self.los_data['time']),
                     len(self.gtlos_data['time']),
                     len(self.gtlos_qiopid_data['time']))

        los_control_input = self.los_data['control_input'][:min_len]
        gtlos_control_input = self.gtlos_data['control_input'][:min_len]

        # 修正：正确进行均匀采样
        indices = np.linspace(0, len(self.gtlos_qiopid_data['control_input'])-1, min_len, dtype=int)
        gtlos_qiopid_control_input = self.gtlos_qiopid_data['control_input'][indices]  # 使用索引采样实际数据
        print(len(self.gtlos_qiopid_data['control_input']),min_len)
        # 处理二维数据，取第一列
        if hasattr(los_control_input, 'shape') and len(los_control_input.shape) > 1:
            los_control_input = los_control_input[:, 0]
            gtlos_control_input = gtlos_control_input[:, 0]
            # 检查gtlos_qiopid_control_input是否为二维
            if hasattr(gtlos_qiopid_control_input, 'shape') and len(gtlos_qiopid_control_input.shape) > 1:
                gtlos_qiopid_control_input = gtlos_qiopid_control_input[:, 0]

        los_mean = np.mean(los_control_input)
        gtlos_mean = np.mean(gtlos_control_input)
        gtlos_qiopid_mean = np.mean(gtlos_qiopid_control_input)

        improvement_gtlos = (los_mean - gtlos_mean) / los_mean * 100 if los_mean != 0 else 0
        improvement_gtlos_qiopid = (los_mean - gtlos_qiopid_mean) / los_mean * 100 if los_mean != 0 else 0

        return improvement_gtlos, improvement_gtlos_qiopid

    def plot_raw_comparison(self):
        """绘制未美化的原始数据对比图，两张图放在一个页面"""
        if not self.los_data or not self.gtlos_data or not self.gtlos_qiopid_data:
            print("请先加载数据")
            return

        # 创建包含两个子图的图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('#f8f9fa')
        ax2.set_facecolor('#f8f9fa')
        # 确保时间向量长度一致
        min_len = min(len(self.los_data['time']),
                     len(self.gtlos_data['time']),
                     len(self.gtlos_qiopid_data['time']))

        time_data = self.los_data['time'][:min_len]
        los_control_input = self.los_data['control_input'][:min_len]
        gtlos_control_input = self.gtlos_data['control_input'][:min_len]

        # 对GTLOS-QIOPID数据进行均匀采样
        indices = np.linspace(0, len(self.gtlos_qiopid_data['control_input'])-1, min_len, dtype=int)
        gtlos_qiopid_control_input = self.gtlos_qiopid_data['control_input'][indices]

        # 计算统计数据
        self.improvement_gtlos, self.improvement_gtlos_qiopid = self._calculate_improvements()

        # 第一张图：使用第一列数据
        if hasattr(los_control_input, 'shape') and len(los_control_input.shape) > 1:
            los_data1 = los_control_input[:, 0]
            gtlos_data1 = gtlos_control_input[:, 0]
            # 检查维度
            if hasattr(gtlos_qiopid_control_input, 'shape') and len(gtlos_qiopid_control_input.shape) > 1:
                gtlos_qiopid_data1 = gtlos_qiopid_control_input[:, 0]
            else:
                gtlos_qiopid_data1 = gtlos_qiopid_control_input
        else:
            los_data1 = los_control_input
            gtlos_data1 = gtlos_control_input
            gtlos_qiopid_data1 = gtlos_qiopid_control_input

        # 第二张图：使用第二列数据（如果是二维数据）
        if (hasattr(los_control_input, 'shape') and len(los_control_input.shape) > 1
            and los_control_input.shape[1] > 1):
            los_data2 = los_control_input[:, 1]
            gtlos_data2 = gtlos_control_input[:, 1]
            if hasattr(gtlos_qiopid_control_input, 'shape') and len(gtlos_qiopid_control_input.shape) > 1:
                gtlos_qiopid_data2 = gtlos_qiopid_control_input[:, 1]
            else:
                gtlos_qiopid_data2 = gtlos_qiopid_control_input
            has_second_column = True
        else:
            # 如果只有一列数据，复制第一列
            los_data2 = los_data1
            gtlos_data2 = gtlos_data1
            gtlos_qiopid_data2 = gtlos_qiopid_data1
            has_second_column = False

        # 绘制第一张图（第一列数据）
        ax1.plot(time_data, los_data1, linewidth=1, color="blue", alpha=0.9, marker='', linestyle='-', label='LOS')
        ax1.plot(time_data, gtlos_data1, linewidth=1, color="red", alpha=0.9, marker='', linestyle='-', label='GTLOS')
        ax1.plot(time_data, gtlos_qiopid_data1, linewidth=1, color='green', alpha=0.9, marker='', linestyle='-', label='GTLOS-QIOPID')

        ax1.set_xlabel('time (s)', fontsize=20)
        ax1.set_ylabel('Control Input-Left', fontsize=20)
        # ax1.set_title('三个算法控制输入对比 ', fontsize=20)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # 绘制第二张图（第二列数据）
        if has_second_column:
            ax2.plot(time_data, los_data2, color='blue', linewidth=1, label='LOS')
            ax2.plot(time_data, gtlos_data2, color='red', linewidth=1, label='GTLOS')
            ax2.plot(time_data, gtlos_qiopid_data2, color='green', linewidth=1, label='GTLOS-QIOPID')

            ax2.set_xlabel('time (s)', fontsize=20)
            ax2.set_ylabel('Control Input-Right', fontsize=20)
            # ax2.set_title('三个算法控制输入对比 ', fontsize=20)
        else:
            ax2.plot(time_data, los_data2, color='blue', linewidth=1, label='LOS')
            ax2.plot(time_data, gtlos_data2, color='red', linewidth=1, label='GTLOS')
            ax2.plot(time_data, gtlos_qiopid_data2, color='green', linewidth=1, label='GTLOS-QIOPID')

            ax2.set_xlabel('时间 (s)', fontsize=20)
            ax2.set_ylabel('控制输入', fontsize=20)
            ax2.set_title('三个算法控制输入对比', fontsize=20)

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

        # 计算并显示统计数据
        los_mean1 = np.mean(los_data1)
        gtlos_mean1 = np.mean(gtlos_data1)
        gtlos_qiopid_mean1 = np.mean(gtlos_qiopid_data1)

        improvement_gtlos1 = (los_mean1 - gtlos_mean1) / los_mean1 * 100 if los_mean1 != 0 else 0
        improvement_gtlos_qiopid1 = (los_mean1 - gtlos_qiopid_mean1) / los_mean1 * 100 if los_mean1 != 0 else 0

#         # 添加统计信息文本框
#         stats_text1 = f'''第一列统计信息:
# LOS平均: {los_mean1:.3f}
# GTLOS平均: {gtlos_mean1:.3f} (改进: {improvement_gtlos1:.1f}%)
# GTLOS-QIOPID平均: {gtlos_qiopid_mean1:.3f} (改进: {improvement_gtlos_qiopid1:.1f}%)'''

#         ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, fontsize=10,
#                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

#         if has_second_column:
#             los_mean2 = np.mean(los_data2)
#             gtlos_mean2 = np.mean(gtlos_data2)
#             gtlos_qiopid_mean2 = np.mean(gtlos_qiopid_data2)

#             improvement_gtlos2 = (los_mean2 - gtlos_mean2) / los_mean2 * 100 if los_mean2 != 0 else 0
#             improvement_gtlos_qiopid2 = (los_mean2 - gtlos_qiopid_mean2) / los_mean2 * 100 if los_mean2 != 0 else 0

#             stats_text2 = f'''第二列统计信息:
# LOS平均: {los_mean2:.3f}
# GTLOS平均: {gtlos_mean2:.3f} (改进: {improvement_gtlos2:.1f}%)
# GTLOS-QIOPID平均: {gtlos_qiopid_mean2:.3f} (改进: {improvement_gtlos_qiopid2:.1f}%)'''

#             ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, fontsize=10,
#                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 优化布局
        plt.tight_layout()

        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'raw_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

        print("原始数据对比图已生成完成！")
        print(f"第一列 - LOS平均: {los_mean1:.3f}, GTLOS平均: {gtlos_mean1:.3f} (改进: {improvement_gtlos1:.1f}%), "
              f"GTLOS-QIOPID平均: {gtlos_qiopid_mean1:.3f} (改进: {improvement_gtlos_qiopid1:.1f}%)")

        # if has_second_column:
        #     print(f"第二列 - LOS平均: {los_mean2:.3f}, GTLOS平均: {gtlos_mean2:.3f} (改进: {improvement_gtlos2:.1f}%), "
        #           f"GTLOS-QIOPID平均: {gtlos_qiopid_mean2:.3f} (改进: {improvement_gtlos_qiopid2:.1f}%)")

        return fig


# 主程序
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = ThreeAlgorithmAnalyzer()

    # 指定三个算法的数据文件路径
    los_data_file = "data\los_simulation_data_20250922_204913.pkl"  # LOS算法数据
    gtlos_data_file = "data2\gtlos_simulation_data_20251010_200332.pkl"  # GTLOS算法数据
    gtlos_qiopid_data_file = "data2\gtlos-qiopid_simulation_data_20251010_200935.pkl"  # GTLOS-QIOPID算法数据

    # 加载数据
    if analyzer.load_data(los_data_file, gtlos_data_file, gtlos_qiopid_data_file):
        # 绘制原始数据对比图
        analyzer.plot_raw_comparison()
        print("所有对比图表已生成完成！")
    else:
        print("数据加载失败，请检查文件路径是否正确。")
