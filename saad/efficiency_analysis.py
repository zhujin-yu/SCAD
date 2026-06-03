import json
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import argparse


def parse_training_log(log_file_path):
    """
    解析训练日志文件，提取效率统计信息
    """
    scene_data = defaultdict(lambda: {'steps': [], 'time': [], 'count': 0})
    total_stats = {'samples': 0, 'total_time': 0, 'total_steps': 0}

    # 尝试多种编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(log_file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except UnicodeDecodeError:
            continue
    else:
        # 如果所有编码都失败，使用二进制模式并忽略错误
        with open(log_file_path, 'rb') as f:
            content = f.read()
        lines = content.decode('utf-8', errors='ignore').splitlines()
        print("使用二进制模式读取文件，忽略编码错误")

    # 正则表达式匹配模式
    scene_pattern = re.compile(r"Scene: (\w+), Timesteps: (\d+), Steps Saved: (-?\d+)")
    efficiency_pattern = re.compile(r"平均步数: ([\d.]+).*平均时间: ([\d.]+)s")
    for line in lines:
        # 匹配场景信息
        scene_match = scene_pattern.search(line)
        if scene_match:
            scene_type = scene_match.group(1)
            timesteps = int(scene_match.group(2))
            steps_saved = int(scene_match.group(3))

            # 过滤无效的步数数据（确保为正数）
            if timesteps > 0:
                scene_data[scene_type]['steps'].append(timesteps)
                scene_data[scene_type]['count'] += 1

        # 匹配效率统计
        efficiency_match = efficiency_pattern.search(line)
        if efficiency_match:
            avg_steps = float(efficiency_match.group(1))
            avg_time = float(efficiency_match.group(2))

    return scene_data, total_stats


def parse_validation_log(log_file_path):
    """
    专门解析验证效率统计报告
    """
    scene_data = defaultdict(lambda: {'steps': [], 'time': [], 'count': 0})

    with open(log_file_path, 'r', encoding='gbk') as f:
        content = f.read()

    # 匹配完整的场景报告块
    scene_blocks = re.findall(
        r'(\w+)场景:\s*\n\s*样本数: (\d+)\s*\n\s*平均步数: ([\d.]+) 步.*?平均PSNR: ([\d.]+), 平均SSIM: ([\d.]+)',
        content,
        re.DOTALL
    )

    for scene_type, count_str, steps_str, psnr_str, ssim_str in scene_blocks:
        count = int(count_str)
        avg_steps = float(steps_str)
        avg_psnr = float(psnr_str)
        avg_ssim = float(ssim_str)

        # 为每个样本添加步数数据
        for _ in range(count):
            scene_data[scene_type]['steps'].append(avg_steps)
        scene_data[scene_type]['count'] = count
        scene_data[scene_type]['psnr'] = avg_psnr
        scene_data[scene_type]['ssim'] = avg_ssim

    return scene_data

def analyze_efficiency(scene_data, total_stats):
    """
    分析效率数据并生成报告
    """
    print("=" * 50)
    print("          场景自适应效率分析报告")
    print("=" * 50)

    total_samples = 0
    total_avg_steps = 0
    total_avg_time = 0

    for scene_type, data in scene_data.items():
        if data['count'] > 0:
            # 确保 steps 列表不为空且避免除零
            if data['steps']:
                avg_steps = np.mean(data['steps'])
            else:
                avg_steps = 0  # 无有效步数数据时的默认值

            # 计算每步时间时添加除零保护
            if avg_steps > 0:
                avg_time_est = avg_steps * 0.05  # 假设每步0.05秒
            else:
                avg_time_est = 0  # 步数为0时时间也为0

            # 计算节省时同样添加保护
            fixed_steps = 20
            steps_saved = max(0, fixed_steps - avg_steps)  # 避免负数
            if fixed_steps > 0:
                efficiency_gain = (steps_saved / fixed_steps) * 100
            else:
                efficiency_gain = 0

            # 时间节省计算
            time_saved_per_sample = steps_saved * 0.05 if avg_steps > 0 else 0

            # 累积总样本数和总步数（用于后续总体统计）
            total_samples += data['count']
            total_avg_steps += avg_steps * data['count']
            total_avg_time += avg_time_est * data['count']

            print(f"      场景: {scene_type.upper()}")
            print(f"      样本数量: {data['count']}")
            print(f"      平均采样步数: {avg_steps:.1f} 步")
            print(f"      估计平均时间: {avg_time_est:.3f} 秒")
            print(f"      步数节省: {steps_saved:.1f} 步 ({efficiency_gain:.1f}%)")
            print(f"      时间节省/样本: {time_saved_per_sample:.3f} 秒")

    # 总体统计添加除零保护
    if total_samples > 0:
        total_avg_steps /= total_samples
        total_avg_time /= total_samples

        # 防止 fixed_steps 为零导致的除零
        fixed_steps = 20
        if fixed_steps <= 0:
            print(" 警告: 基准步数配置无效，无法计算总体效率提升")
            return

        overall_steps_saved = max(0, fixed_steps - total_avg_steps)
        overall_efficiency_gain = (overall_steps_saved / fixed_steps) * 100
        overall_time_saving = overall_steps_saved * 0.05

        print(f"\n" + "=" * 50)
        print(f"      总体统计 (基于 {total_samples} 个样本)")
        print(f"      平均采样步数: {total_avg_steps:.1f} 步")
        print(f"      总体效率提升: {overall_efficiency_gain:.1f}%")
        print(f"      平均时间节省/样本: {overall_time_saving:.3f} 秒")
        print(f"      总时间节省: {overall_time_saving * total_samples:.1f} 秒")

        # 业务场景估算
        daily_volumes = [1000, 5000, 10000]
        print(f"  业务场景估算:")
        for daily_volume in daily_volumes:
            daily_time_saving = overall_time_saving * daily_volume
            print(f"   每日 {daily_volume} 样本: 节省 {daily_time_saving / 3600:.2f} 小时")
    else:
        print("  警告: 没有有效的样本数据用于总体统计分析")


def plot_efficiency_comparison(scene_data):
    """
    绘制效率对比图表
    """
    # 过滤无效数据
    valid_data = {
        scene: data for scene, data in scene_data.items()
        if data['count'] > 0 and data['steps']  # 确保有样本且有有效步数
    }

    if not valid_data:
        print("没有足够的数据来生成图表")
        return

    scenes = []
    avg_steps = []
    efficiency_gains = []
    fixed_steps = 20

    for scene_type, data in valid_data.items():
        steps = np.mean(data['steps'])
        scenes.append(scene_type.upper())  # 使用大写英文
        avg_steps.append(steps)

        # 计算效率提升时添加除零保护
        if fixed_steps > 0:
            gain = (max(0, fixed_steps - steps) / fixed_steps) * 100
        else:
            gain = 0
        efficiency_gains.append(gain)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1：平均步数对比
    x_pos = np.arange(len(scenes))
    bars1 = ax1.bar(x_pos, avg_steps, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scenes)
    ax1.set_ylabel('Average Sampling Steps')
    ax1.set_title('Average Sampling Steps by Scene')
    ax1.axhline(y=fixed_steps, color='r', linestyle='--', alpha=0.7, label=f'Fixed {fixed_steps} Steps Baseline')
    ax1.legend()

    # 在柱状图上添加数值标签
    for i, v in enumerate(avg_steps):
        ax1.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')

    # 子图2：效率提升百分比
    bars2 = ax2.bar(x_pos, efficiency_gains, color=['#ff9ff3', '#f368e0', '#ff9f43', '#ee5253'])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenes)
    ax2.set_ylabel('Efficiency Gain (%)')
    ax2.set_title('Efficiency Improvement by Scene')

    # 在柱状图上添加数值标签
    for i, v in enumerate(efficiency_gains):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('efficiency_analysis.png', dpi=300, bbox_inches='tight')
    print(f"图表已保存为: efficiency_analysis.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='分析场景自适应采样效率')
    parser.add_argument('--log-file', type=str, default='training_log.txt',
                        help='训练日志文件路径 (默认: training_log.txt)')
    parser.add_argument('--plot', action='store_true',
                        help='生成效率对比图表')

    args = parser.parse_args()

    try:
        scene_data, total_stats = parse_training_log(args.log_file)
        analyze_efficiency(scene_data, total_stats)

        if args.plot:
            plot_efficiency_comparison(scene_data)

    except FileNotFoundError:
        print(f"   错误: 找不到日志文件 {args.log_file}")
        print("请确保:")
        print("1. 日志文件存在且路径正确")
        print("2. 日志中包含场景自适应采样信息")
        print("3. 或者使用 --log-file 参数指定正确的日志文件路径")
    except Exception as e:
        print(f"   分析过程出错: {str(e)}")


if __name__ == "__main__":
    main()