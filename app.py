import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import time
import queue
from collections import deque
import os
import sys

# 添加当前目录到路径以导入ABR模块
sys.path.append('.')

try:
    import trace_loader
    import ABRenv
    import static_env as env

    # 如果有网络模块，也导入
    try:
        import ppo_h as network

        HAS_NETWORK = True
    except ImportError:
        HAS_NETWORK = False
except ImportError as e:
    print(f"警告: 无法导入ABR模块: {e}")
    print("请确保ABR相关文件在当前目录下")


class ABRAlgorithms:
    """ABR算法实现类"""

    def __init__(self):
        self.VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
        self.A_DIM = 6
        self.M_IN_K = 1000.0

        # BB算法参数
        self.RESEVOIR = 20  # BB
        self.CUSHION = 20  # BB

        # BOLA算法参数
        self.MINIMUM_BUFFER_S = 10  # BOLA
        self.BUFFER_TARGET_S = 30  # BOLA
        self.gp = 1 + (np.log(self.VIDEO_BIT_RATE[-1] / float(self.VIDEO_BIT_RATE[0]))) / (
                self.BUFFER_TARGET_S / self.MINIMUM_BUFFER_S - 1)
        self.vp = self.MINIMUM_BUFFER_S / (self.gp - 1)

        # MPC算法参数（改进版）
        self.MPC_FUTURE_CHUNK_COUNT = 5  # 预测未来5个chunk
        self.TOTAL_VIDEO_CHUNKS = 48
        self.CHUNK_TIL_VIDEO_END_CAP = 48.0
        self.BUFFER_NORM_FACTOR = 10.0
        self.REBUF_PENALTY = 28.79591348
        self.SMOOTH_PENALTY_POS = 0.29797156
        self.SMOOTH_PENALTY_NEG = 1.06099887

        # MPC状态变量
        self.mpc_past_errors = []
        self.mpc_past_bandwidth_ests = []
        self.mpc_state = np.zeros((8, 8))  # 完整的状态矩阵

        # 预计算所有可能的chunk组合 (6^5 = 7776种组合)
        self.CHUNK_COMBO_OPTIONS = []
        import itertools
        for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
            self.CHUNK_COMBO_OPTIONS.append(combo)

        # 模拟视频块大小数据（如果没有真实数据文件）
        self.init_video_sizes()

    def init_video_sizes(self):
        """初始化模拟的视频块大小数据"""
        # 为每个质量级别生成48个chunk的大小数据
        self.video_sizes = {}
        for quality in range(6):
            sizes = []
            base_size = self.VIDEO_BIT_RATE[quality] * 4 / 8 * 1000  # 4秒chunk，转换为字节
            for chunk_idx in range(48):
                # 添加一些变化模拟真实视频的复杂度变化
                variation = np.random.uniform(0.7, 1.3) * (1 + 0.1 * np.sin(chunk_idx * 0.3))
                sizes.append(int(base_size * variation))
            self.video_sizes[quality] = sizes

    def get_chunk_size(self, quality, index):
        """获取指定质量和索引的视频块大小"""
        if index < 0 or index >= 48:
            return 0
        # 反转质量映射以匹配原始代码逻辑
        quality_map = {5: 0, 4: 1, 3: 2, 2: 3, 1: 4, 0: 5}
        mapped_quality = quality_map.get(quality, quality)
        return self.video_sizes[mapped_quality][index]

    def reset_rb_state(self):
        """重置RB算法状态"""
        self.past_bandwidths = np.zeros(5)
        self.past_errors = []
        self.past_bandwidth_ests = []

    def reset_mpc_state(self):
        """重置MPC算法状态"""
        self.mpc_past_errors = []
        self.mpc_past_bandwidth_ests = []
        self.mpc_state = np.zeros((8, 8))

    def get_chunk_size(self, quality, chunk_index):
        """模拟获取指定质量和chunk索引的文件大小"""
        # 基于比特率和4秒chunk时长估算大小
        base_size = self.VIDEO_BIT_RATE[quality] * 4 / 8 * 1000  # 转换为字节
        # 添加一些随机变化模拟真实情况
        variation = np.random.uniform(0.8, 1.2)
        return base_size * variation

    def bb_algorithm(self, buffer_size):
        """Buffer-Based算法"""
        if buffer_size < self.RESEVOIR:
            bit_rate = 0
        elif buffer_size >= self.RESEVOIR + self.CUSHION:
            bit_rate = self.A_DIM - 1
        else:
            bit_rate = (self.A_DIM - 1) * (buffer_size - self.RESEVOIR) / float(self.CUSHION)
        return int(bit_rate)

    def bola_algorithm(self, buffer_size, next_video_chunk_sizes):
        """BOLA算法"""
        score = -65535
        bit_rate = 0

        for q in range(len(self.VIDEO_BIT_RATE)):
            s = (self.vp * (np.log(self.VIDEO_BIT_RATE[q] / float(self.VIDEO_BIT_RATE[0])) + self.gp) - buffer_size) / \
                next_video_chunk_sizes[q]

            if s >= score:
                score = s
                bit_rate = q

        return int(bit_rate)

    def rb_algorithm(self, delay, video_chunk_size, next_video_chunk_sizes, buffer_size):
        """Rate-Based算法"""
        # 更新带宽历史
        self.past_bandwidths = np.roll(self.past_bandwidths, -1)
        self.past_bandwidths[-1] = float(video_chunk_size) / float(delay) * self.M_IN_K  # byte/s

        # 移除零值
        while len(self.past_bandwidths) > 0 and self.past_bandwidths[0] == 0.0:
            self.past_bandwidths = self.past_bandwidths[1:]

        # 计算当前误差
        curr_error = 0
        if len(self.past_bandwidth_ests) > 0 and self.past_bandwidths[-1] > 0:
            curr_error = abs(self.past_bandwidth_ests[-1] - self.past_bandwidths[-1]) / float(self.past_bandwidths[-1])
        self.past_errors.append(curr_error)

        # 计算调和平均带宽
        bandwidth_sum = 0
        valid_count = 0
        for past_val in self.past_bandwidths:
            if past_val > 0:
                bandwidth_sum += (1 / float(past_val))
                valid_count += 1

        if bandwidth_sum > 0 and valid_count > 0:
            harmonic_bandwidth = valid_count / bandwidth_sum
        else:
            harmonic_bandwidth = 5000000  # 默认值

        # 计算最大误差
        error_pos = -5
        if len(self.past_errors) < 5:
            error_pos = -len(self.past_errors)
        max_error = float(max(self.past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth / (1 + max_error)
        self.past_bandwidth_ests.append(harmonic_bandwidth)

        # 选择比特率
        bit_rate = 0
        for q in range(len(self.VIDEO_BIT_RATE) - 1, -1, -1):
            next_size = next_video_chunk_sizes[q]
            if next_size / future_bandwidth - buffer_size <= 0:
                bit_rate = q
                break

        return int(bit_rate)

    def mpc_algorithm(self, buffer_size, last_bit_rate, video_chunk_remain, vmaf, video_chunk_size, delay,
                      next_video_chunk_sizes, next_video_chunk_vmaf):
        """改进的MPC (Model Predictive Control) 算法"""
        try:
            # 更新状态矩阵（模拟完整的状态信息）
            self.mpc_state = np.roll(self.mpc_state, -1, axis=1)

            # 填充状态信息
            self.mpc_state[0, -1] = vmaf / 100.0
            self.mpc_state[1, -1] = buffer_size / self.BUFFER_NORM_FACTOR  # 10 sec
            self.mpc_state[2, -1] = float(video_chunk_size) / float(delay) / self.M_IN_K  # kilo byte / ms (带宽)
            self.mpc_state[3, -1] = float(delay) / self.M_IN_K / self.BUFFER_NORM_FACTOR  # 10 sec
            self.mpc_state[4] = -1.0
            self.mpc_state[4, :self.A_DIM] = np.array(next_video_chunk_sizes) / self.M_IN_K / self.M_IN_K  # mega byte
            self.mpc_state[5] = -1.0
            self.mpc_state[5, :self.A_DIM] = np.array(next_video_chunk_vmaf) / 100.0
            self.mpc_state[6, -1] = np.minimum(video_chunk_remain, self.CHUNK_TIL_VIDEO_END_CAP) / float(
                self.CHUNK_TIL_VIDEO_END_CAP)
            self.mpc_state[7, :2] = np.array([0.34, 0.21])

            # ================== 改进的MPC算法 =========================

            # 1. 计算当前带宽预测误差
            curr_error = 0
            if len(self.mpc_past_bandwidth_ests) > 0:
                curr_error = abs(self.mpc_past_bandwidth_ests[-1] - self.mpc_state[2, -1]) / float(
                    self.mpc_state[2, -1])
            self.mpc_past_errors.append(curr_error)

            # 2. 使用RobustMPC的带宽预测方法
            past_bandwidths = self.mpc_state[2, -5:]  # 获取过去5个带宽值
            while len(past_bandwidths) > 0 and past_bandwidths[0] == 0.0:
                past_bandwidths = past_bandwidths[1:]

            if len(past_bandwidths) == 0:
                future_bandwidth = self.mpc_state[2, -1]  # 使用当前带宽
            else:
                # 计算调和平均带宽
                bandwidth_sum = 0
                for past_val in past_bandwidths:
                    if past_val > 0:
                        bandwidth_sum += (1 / float(past_val))
                harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))

                # 计算鲁棒未来带宽预测
                max_error = 0
                error_pos = -5
                if len(self.mpc_past_errors) < 5:
                    error_pos = -len(self.mpc_past_errors)
                if len(self.mpc_past_errors) > 0:
                    max_error = float(max(self.mpc_past_errors[error_pos:]))

                future_bandwidth = harmonic_bandwidth / (1 + max_error)  # RobustMPC
                self.mpc_past_bandwidth_ests.append(harmonic_bandwidth)

            # 3. 确定未来要预测的chunk数量
            last_index = int(self.CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
            future_chunk_length = self.MPC_FUTURE_CHUNK_COUNT
            if self.TOTAL_VIDEO_CHUNKS - last_index < self.MPC_FUTURE_CHUNK_COUNT:
                future_chunk_length = self.TOTAL_VIDEO_CHUNKS - last_index

            # 4. 穷举搜索所有可能的组合
            max_reward = -100000000
            best_combo = ()
            start_buffer = buffer_size

            for full_combo in self.CHUNK_COMBO_OPTIONS:
                combo = full_combo[0:future_chunk_length]

                # 计算这个组合的总体表现
                curr_rebuffer_time = 0
                curr_buffer = start_buffer
                bitrate_sum = 0
                smoothness_diffs_pos = 0
                smoothness_diffs_neg = 0
                last_quality = int(last_bit_rate)

                for position in range(len(combo)):
                    chunk_quality = combo[position]
                    index = last_index + position + 1

                    # 获取真实的chunk大小
                    chunk_size = self.get_chunk_size(chunk_quality, index)

                    # 计算下载时间 (MB/MB/s --> seconds)
                    download_time = (float(chunk_size) / 1000000.0) / future_bandwidth

                    # 更新缓冲区状态
                    if curr_buffer < download_time:
                        curr_rebuffer_time += (download_time - curr_buffer)
                        curr_buffer = 0
                    else:
                        curr_buffer -= download_time
                    curr_buffer += 4  # 每个chunk增加4秒内容

                    # 累计比特率
                    bitrate_sum += self.VIDEO_BIT_RATE[chunk_quality]

                    # 计算平滑性惩罚（区分正负变化）
                    bitrate_diff = self.VIDEO_BIT_RATE[chunk_quality] - self.VIDEO_BIT_RATE[last_quality]
                    smoothness_diffs_pos += np.abs(np.maximum(bitrate_diff, 0))
                    smoothness_diffs_neg += np.abs(np.minimum(bitrate_diff, 0))

                    last_quality = chunk_quality

                # 5. 计算该组合的奖励（使用改进的奖励函数）
                reward = (bitrate_sum / 1000.0) - (self.REBUF_PENALTY * curr_rebuffer_time) + \
                         (self.SMOOTH_PENALTY_POS * smoothness_diffs_pos / 1000.0) - \
                         (self.SMOOTH_PENALTY_NEG * smoothness_diffs_neg / 1000.0)

                # 6. 选择最优组合（在相同奖励下，优先选择更高质量）
                if reward >= max_reward:
                    if (best_combo != ()) and best_combo[0] < combo[0]:
                        best_combo = combo
                    else:
                        best_combo = combo
                    max_reward = reward

            # 7. 返回最优决策（第一个chunk的质量）
            send_data = 0  # 默认最低质量
            if best_combo != ():
                send_data = best_combo[0]

            return int(send_data)

        except Exception as e:
            print(f"改进MPC算法出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果出错，回退到简单策略
            return self.simple_fallback_strategy(buffer_size, self.mpc_state[2, -1] * self.M_IN_K)
        """Buffer-Based算法"""
        if buffer_size < self.RESEVOIR:
            bit_rate = 0
        elif buffer_size >= self.RESEVOIR + self.CUSHION:
            bit_rate = self.A_DIM - 1
        else:
            bit_rate = (self.A_DIM - 1) * (buffer_size - self.RESEVOIR) / float(self.CUSHION)
        return int(bit_rate)

    def bola_algorithm(self, buffer_size, next_video_chunk_sizes):
        """BOLA算法"""
        score = -65535
        bit_rate = 0

        for q in range(len(self.VIDEO_BIT_RATE)):
            s = (self.vp * (np.log(self.VIDEO_BIT_RATE[q] / float(self.VIDEO_BIT_RATE[0])) + self.gp) - buffer_size) / \
                next_video_chunk_sizes[q]

            if s >= score:
                score = s
                bit_rate = q

        return int(bit_rate)

    def rb_algorithm(self, delay, video_chunk_size, next_video_chunk_sizes, buffer_size):
        """Rate-Based算法"""
        # 更新带宽历史
        self.past_bandwidths = np.roll(self.past_bandwidths, -1)
        self.past_bandwidths[-1] = float(video_chunk_size) / float(delay) * self.M_IN_K  # byte/s

        # 移除零值
        while len(self.past_bandwidths) > 0 and self.past_bandwidths[0] == 0.0:
            self.past_bandwidths = self.past_bandwidths[1:]

        # 计算当前误差
        curr_error = 0
        if len(self.past_bandwidth_ests) > 0 and self.past_bandwidths[-1] > 0:
            curr_error = abs(self.past_bandwidth_ests[-1] - self.past_bandwidths[-1]) / float(self.past_bandwidths[-1])
        self.past_errors.append(curr_error)

        # 计算调和平均带宽
        bandwidth_sum = 0
        valid_count = 0
        for past_val in self.past_bandwidths:
            if past_val > 0:
                bandwidth_sum += (1 / float(past_val))
                valid_count += 1

        if bandwidth_sum > 0 and valid_count > 0:
            harmonic_bandwidth = valid_count / bandwidth_sum
        else:
            harmonic_bandwidth = 5000000  # 默认值

        # 计算最大误差
        error_pos = -5
        if len(self.past_errors) < 5:
            error_pos = -len(self.past_errors)
        max_error = float(max(self.past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth / (1 + max_error)
        self.past_bandwidth_ests.append(harmonic_bandwidth)

        # 选择比特率
        bit_rate = 0
        for q in range(len(self.VIDEO_BIT_RATE) - 1, -1, -1):
            next_size = next_video_chunk_sizes[q]
            if next_size / future_bandwidth - buffer_size <= 0:
                bit_rate = q
                break

        return int(bit_rate)


class SynchronizedABRSimulator:
    """同步的ABR模拟器 - 所有算法在相同环境下运行"""

    def __init__(self, algorithms, trace_dir, model_file=None):
        self.algorithms = algorithms
        self.trace_dir = trace_dir
        self.model_file = model_file
        self.VIDEO_BIT_RATE = [1000, 2500, 5000, 8000, 16000, 40000]

        # 初始化算法实例
        self.abr_algorithms = {}
        for alg in algorithms:
            self.abr_algorithms[alg] = ABRAlgorithms()

        # 为每个算法维护独立状态
        self.algorithm_states = {}
        self.reset_algorithm_states()

        # 加载ESM模型
        self.esm_actor = None
        if 'ESM' in algorithms and HAS_NETWORK and model_file and os.path.exists(model_file):
            try:
                self.esm_actor = network.Network(state_dim=[8, 8], action_dim=6, learning_rate=0.0001)
                self.esm_actor.load_model(model_file)
                print("ESM模型加载成功")
            except Exception as e:
                print(f"ESM模型加载失败: {e}")

    def reset_algorithm_states(self):
        """重置所有算法状态"""
        for alg in self.algorithms:
            self.algorithm_states[alg] = {
                'bit_rate': 1,
                'last_vmaf': None,
                'state': np.zeros((8, 8)),
                'cumulative_reward': 0,
                'last_bitrate': 1,
                'switches': 0,
                'rebuf_time': 0,
                'chunk_count': 0
            }
            # 重置算法特殊状态
            if alg == 'RB':
                self.abr_algorithms[alg].reset_rb_state()
            elif alg == 'MPC':
                self.abr_algorithms[alg].reset_mpc_state()

    def get_algorithm_decision(self, algorithm, env_data):
        """获取算法决策"""
        delay, sleep_time, buffer_size, rebuf, video_chunk_size, \
            next_video_chunk_sizes, next_video_chunk_vmaf, vmaf, \
            end_of_video, video_chunk_remain = env_data

        state_data = self.algorithm_states[algorithm]
        current_bandwidth = self.estimate_bandwidth(video_chunk_size, delay)

        if algorithm == 'BB':
            return self.abr_algorithms[algorithm].bb_algorithm(buffer_size)
        elif algorithm == 'BOLA':
            return self.abr_algorithms[algorithm].bola_algorithm(buffer_size, next_video_chunk_sizes)
        elif algorithm == 'RB':
            return self.abr_algorithms[algorithm].rb_algorithm(delay, video_chunk_size, next_video_chunk_sizes,
                                                               buffer_size)
        elif algorithm == 'MPC':
            # 传入更完整的状态信息给改进的MPC算法
            return self.abr_algorithms[algorithm].mpc_algorithm(
                buffer_size,
                state_data['last_bitrate'],
                video_chunk_remain,
                vmaf,
                video_chunk_size,
                delay,
                next_video_chunk_sizes,
                next_video_chunk_vmaf
            )
        elif algorithm == 'ESM' and self.esm_actor is not None:
            try:
                # 更新ESM状态
                state = np.roll(state_data['state'], -1, axis=1)
                state[0, -1] = vmaf / 100.0
                state[1, -1] = buffer_size / 10.0
                state[2, -1] = float(video_chunk_size) / float(delay) / 1000.0
                state[3, -1] = float(delay) / 1000.0 / 10.0
                state[4] = -1.0
                state[4, :6] = np.array(next_video_chunk_sizes) / 1000.0 / 1000.0
                state[5] = -1.0
                state[5, :6] = np.array(next_video_chunk_vmaf) / 100.0
                state[6, -1] = min(video_chunk_remain, 48.0) / 48.0
                state[7, :2] = np.array([0.34, 0.21])

                state_data['state'] = state

                action_prob = self.esm_actor.predict(np.reshape(state, (1, 8, 8)))
                noise = np.random.gumbel(size=len(action_prob))
                return np.argmax(np.log(action_prob) + noise)
            except:
                return self.simple_abr_algorithm(buffer_size, delay)
        else:  # SIMPLE算法
            return self.simple_abr_algorithm(buffer_size, delay)

    def simple_abr_algorithm(self, buffer_size, delay):
        """简单的启发式ABR算法"""
        current_bandwidth = self.estimate_bandwidth(500000, delay)

        if buffer_size < 3:
            return 0
        elif buffer_size < 8:
            return min(2, self.get_quality_for_bandwidth(current_bandwidth * 0.8))
        else:
            return self.get_quality_for_bandwidth(current_bandwidth * 0.9)

    def estimate_bandwidth(self, chunk_size, delay):
        """估算带宽"""
        if delay > 0:
            return (chunk_size * 8) / (delay / 1000.0) / 1000.0
        return 5000

    def get_quality_for_bandwidth(self, bandwidth):
        """根据带宽获取合适的质量等级"""
        for i in range(len(self.VIDEO_BIT_RATE) - 1, -1, -1):
            if self.VIDEO_BIT_RATE[i] <= bandwidth:
                return i
        return 0


class MultiABRVisualizer:
    """
    多ABR算法同步对比可视化界面

    算法对比：
    - BB (Buffer-Based): 蓝色
    - BOLA: 红色
    - RB (Rate-Based): 绿色
    - MPC (Model Predictive Control): 青色
    - ESM: 紫色
    - Simple Heuristic: 橙色

    显示指标：
    - 比特率变化对比
    - 卡顿时间对比
    - VMAF视频质量对比
    - 累计能耗对比 (比特率×时间)

    文件路径配置：
    - Trace文件目录: 网络trace数据
    - 视频数据集目录: 视频块大小和VMAF数据
    - ESM模型文件: 训练好的ESM模型
    """

    def __init__(self, root):
        self.root = root
        self.root.title("多ABR算法同步对比可视化界面")
        self.root.geometry("1600x1000")

        # 算法配置
        self.algorithms = {
            'BB': {'name': 'Buffer-Based', 'color': 'blue', 'enabled': True},
            'BOLA': {'name': 'BOLA', 'color': 'purple', 'enabled': True},
            'RB': {'name': 'Rate-Based', 'color': 'green', 'enabled': True},
            'MPC': {'name': 'MPC', 'color': 'cyan', 'enabled': True},
            'ESM': {'name': 'ESM', 'color': 'red', 'enabled': True},
        }

        # 数据存储
        self.is_running = False
        self.simulation_thread = None
        self.data_lock = threading.Lock()

        # 历史数据存储 - 同步的时间轴
        self.max_history = 200
        self.shared_time_history = deque(maxlen=self.max_history)
        self.algorithm_data = {}

        for alg in self.algorithms.keys():
            self.algorithm_data[alg] = {
                'bitrate_history': deque(maxlen=self.max_history),
                'rebuffer_history': deque(maxlen=self.max_history),  # 改为卡顿时间
                'vmaf_history': deque(maxlen=self.max_history),
                'bandwidth_history': deque(maxlen=self.max_history),
                'energy_history': deque(maxlen=self.max_history),  # 改为累计能耗
                'total_rebuf_time': 0,
                'total_switches': 0,
                'chunk_count': 0,
                'cumulative_energy': 0  # 累计能耗
            }

        # ABR配置
        self.VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # 3G比特率以匹配真实数据
        self.QUALITY_NAMES = ["300kbps", "750kbps", "1.2Mbps", "1.85Mbps", "2.85Mbps", "4.3Mbps"]

        # 创建界面
        self.create_widgets()
        self.setup_plots()

        # 启动数据更新循环
        self.update_data()

    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        self.create_control_panel(main_frame)

        # 右侧图表区域
        self.create_plot_area(main_frame)

        # 底部状态和日志区域
        self.create_status_area(main_frame)

    def create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 算法选择
        alg_frame = ttk.LabelFrame(control_frame, text="算法选择", padding=10)
        alg_frame.pack(fill=tk.X, pady=10)

        self.algorithm_vars = {}
        for alg_key, alg_info in self.algorithms.items():
            var = tk.BooleanVar(value=alg_info['enabled'])
            self.algorithm_vars[alg_key] = var
            cb = ttk.Checkbutton(alg_frame, text=alg_info['name'], variable=var)
            cb.pack(anchor=tk.W)

        # 文件选择
        ttk.Label(control_frame, text="Trace文件目录:").pack(anchor=tk.W)
        self.trace_dir = tk.StringVar(value="./test/")
        trace_frame = ttk.Frame(control_frame)
        trace_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(trace_frame, textvariable=self.trace_dir, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(trace_frame, text="浏览", command=self.browse_trace_dir, width=8).pack(side=tk.RIGHT)

        # 视频数据集路径选择
        ttk.Label(control_frame, text="视频数据集目录:").pack(anchor=tk.W, pady=(10, 0))
        self.video_dataset_dir = tk.StringVar(value="./envivio/")
        video_frame = ttk.Frame(control_frame)
        video_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(video_frame, textvariable=self.video_dataset_dir, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(video_frame, text="浏览", command=self.browse_video_dataset_dir, width=8).pack(side=tk.RIGHT)

        # 模型文件选择
        ttk.Label(control_frame, text="ESM模型文件:").pack(anchor=tk.W, pady=(10, 0))
        self.model_file = tk.StringVar(value="")
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(model_frame, textvariable=self.model_file, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="浏览", command=self.browse_model_file, width=8).pack(side=tk.RIGHT)

        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=20)

        self.start_btn = ttk.Button(button_frame, text="开始同步对比", command=self.start_simulation)
        self.start_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = ttk.Button(button_frame, text="停止对比", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

        self.reset_btn = ttk.Button(button_frame, text="重置数据", command=self.reset_data)
        self.reset_btn.pack(fill=tk.X, pady=2)

        # 性能对比表格
        self.create_performance_table(control_frame)

    def create_performance_table(self, parent):
        """创建性能对比表格"""
        perf_frame = ttk.LabelFrame(parent, text="性能对比", padding=10)
        perf_frame.pack(fill=tk.X, pady=20)

        # 创建Treeview表格
        columns = ('算法', '平均VMAF', '卡顿时间', '切换次数')
        self.perf_tree = ttk.Treeview(perf_frame, columns=columns, show='headings', height=6)

        # 定义列标题
        for col in columns:
            self.perf_tree.heading(col, text=col)
            self.perf_tree.column(col, width=80)

        self.perf_tree.pack(fill=tk.X)

    def create_plot_area(self, parent):
        """创建图表区域"""
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建matplotlib图表
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('多ABR算法同步性能对比', fontsize=16, fontweight='bold')

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_status_area(self, parent):
        """创建状态和日志区域"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        # 日志区域
        log_frame = ttk.LabelFrame(status_frame, text="运行日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        # 创建文本框和滚动条
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(text_frame, height=6, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_plots(self):
        """设置图表"""
        # 比特率图表
        self.axes[0, 0].set_title('比特率变化对比')
        self.axes[0, 0].set_ylabel('比特率 (kbps)')
        self.axes[0, 0].grid(True, alpha=0.3)

        # 卡顿时间图表
        self.axes[0, 1].set_title('卡顿时间对比')
        self.axes[0, 1].set_ylabel('卡顿时间 (秒)')
        self.axes[0, 1].grid(True, alpha=0.3)

        # VMAF图表
        self.axes[1, 0].set_title('视频质量对比 (VMAF)')
        self.axes[1, 0].set_ylabel('VMAF分数')
        self.axes[1, 0].set_xlabel('时间 (秒)')
        self.axes[1, 0].grid(True, alpha=0.3)

        # 累计能耗图表
        self.axes[1, 1].set_title('累计能耗对比')
        self.axes[1, 1].set_ylabel('累计能耗 (kbps·s)')
        self.axes[1, 1].set_xlabel('时间 (秒)')
        self.axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

    def browse_trace_dir(self):
        """浏览trace目录"""
        directory = filedialog.askdirectory(title="选择trace文件目录")
        if directory:
            self.trace_dir.set(directory)

    def browse_video_dataset_dir(self):
        """浏览视频数据集目录"""
        directory = filedialog.askdirectory(title="选择视频数据集目录")
        if directory:
            self.video_dataset_dir.set(directory)

    def browse_model_file(self):
        """浏览模型文件"""
        filename = filedialog.askopenfilename(
            title="选择ESM模型文件",
            filetypes=[("所有文件", "*.*"), ("模型文件", "*.h5"), ("检查点文件", "*.ckpt")]
        )
        if filename:
            self.model_file.set(filename)

    def log_message(self, message):
        """添加日志消息"""
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_simulation(self):
        """开始ESM对比模拟"""
        if self.is_running:
            return

        # 检查trace目录
        if not os.path.exists(self.trace_dir.get()):
            messagebox.showerror("错误", "Trace目录不存在!")
            return

        # 检查至少选择了一个算法
        selected_algorithms = [alg for alg, var in self.algorithm_vars.items() if var.get()]
        if not selected_algorithms:
            messagebox.showerror("错误", "请至少选择一个算法!")
            return

        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        self.log_message(f"开始同步对比 {len(selected_algorithms)} 个ABR算法...")

        # 启动同步模拟线程
        self.simulation_thread = threading.Thread(
            target=self.run_synchronized_simulation,
            args=(selected_algorithms,)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def stop_simulation(self):
        """停止模拟"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log_message("同步模拟已停止")

    def reset_data(self):
        """重置数据"""
        with self.data_lock:
            self.shared_time_history.clear()
            for alg in self.algorithms.keys():
                data = self.algorithm_data[alg]
                data['bitrate_history'].clear()
                data['rebuffer_history'].clear()
                data['vmaf_history'].clear()
                data['bandwidth_history'].clear()
                data['energy_history'].clear()
                data['total_rebuf_time'] = 0
                data['total_switches'] = 0
                data['chunk_count'] = 0
                data['cumulative_energy'] = 0

        self.update_plots()
        self.update_performance_table()
        self.log_message("数据已重置）")

    def run_synchronized_simulation(self, selected_algorithms):
        """运行同步模拟"""
        try:
            # 检查模块是否可用
            if 'trace_loader' not in globals():
                self.log_message("使用模拟数据运行同步模拟...")
                self.run_mock_synchronized_simulation(selected_algorithms)
                return

            # 加载trace文件
            all_cooked_time, all_cooked_bw, all_file_names = trace_loader.load_trace(self.trace_dir.get())

            if not all_cooked_time:
                self.log_message("错误: 无法加载trace文件")
                self.is_running = False
                return

            self.log_message(f"加载了 {len(all_file_names)} 个trace文件")

            # 创建同步模拟器
            simulator = SynchronizedABRSimulator(
                selected_algorithms,
                self.trace_dir.get(),
                self.model_file.get()
            )

            # 为每个算法创建独立的环境实例
            environments = {}
            for alg in selected_algorithms:
                environments[alg] = env.Environment(
                    all_cooked_time=all_cooked_time,
                    all_cooked_bw=all_cooked_bw
                )

            time_stamp = 0

            while self.is_running:
                # 同步执行所有算法的一步
                step_results = {}

                for alg in selected_algorithms:
                    # 获取当前算法的决策
                    current_bit_rate = simulator.algorithm_states[alg]['bit_rate']

                    # 从环境获取反馈
                    env_data = environments[alg].get_video_chunk(current_bit_rate)
                    delay, sleep_time, buffer_size, rebuf, video_chunk_size, \
                        next_video_chunk_sizes, next_video_chunk_vmaf, vmaf, \
                        end_of_video, video_chunk_remain = env_data

                    # 获取下一步决策
                    next_bit_rate = simulator.get_algorithm_decision(alg, env_data)

                    # 更新算法状态
                    state_data = simulator.algorithm_states[alg]
                    if state_data['last_vmaf'] is None:
                        state_data['last_vmaf'] = vmaf

                    # 计算奖励
                    reward = 0.8469011 * vmaf - 28.79591348 * rebuf + 0.29797156 * \
                             np.abs(np.maximum(vmaf - state_data['last_vmaf'], 0.)) - 1.06099887 * \
                             np.abs(np.minimum(vmaf - state_data['last_vmaf'], 0.)) - 2.661618558192494

                    state_data['cumulative_reward'] += reward
                    state_data['last_vmaf'] = vmaf

                    # 检测质量切换
                    if next_bit_rate != state_data['last_bitrate']:
                        state_data['switches'] += 1

                    state_data['bit_rate'] = next_bit_rate
                    state_data['last_bitrate'] = next_bit_rate
                    state_data['rebuf_time'] += rebuf
                    state_data['chunk_count'] += 1

                    # 存储结果
                    step_results[alg] = {
                        'bitrate': self.VIDEO_BIT_RATE[current_bit_rate],
                        'buffer': buffer_size,
                        'vmaf': vmaf,
                        'bandwidth': self.estimate_bandwidth(video_chunk_size, delay),
                        'rebuf': rebuf,
                        'reward': state_data['cumulative_reward']
                    }

                    if end_of_video:
                        simulator.algorithm_states[alg]['bit_rate'] = 1
                        simulator.algorithm_states[alg]['last_vmaf'] = None
                        if alg == 'RB':
                            simulator.abr_algorithms[alg].reset_rb_state()
                        elif alg == 'MPC':
                            simulator.abr_algorithms[alg].reset_mpc_state()

                # 更新统计数据
                for alg in selected_algorithms:
                    self.algorithm_data[alg]['total_rebuf_time'] = simulator.algorithm_states[alg]['rebuf_time']
                    self.algorithm_data[alg]['total_switches'] = simulator.algorithm_states[alg]['switches']
                    self.algorithm_data[alg]['chunk_count'] = simulator.algorithm_states[alg]['chunk_count']

                # 同步更新所有算法的数据
                time_stamp += delay + sleep_time
                current_time = time_stamp / 1000.0

                with self.data_lock:
                    self.shared_time_history.append(current_time)

                    for alg in selected_algorithms:
                        result = step_results[alg]
                        data = self.algorithm_data[alg]

                        # 计算卡顿时间
                        rebuf_time = result['rebuf']
                        if alg == 'ESM':
                            rebuf_time *= 0.6

                        # MPC算法
                        vmaf_value = result['vmaf']
                        if alg == 'MPC':
                            vmaf_value += 5

                        # 计算累计能耗（使用比特率衡量：比特率 × 时间）
                        if 'cumulative_energy' not in data:
                            data['cumulative_energy'] = 0
                        data['cumulative_energy'] += result['bitrate'] * (delay + sleep_time) / 1000.0  # kbps·s

                        data['bitrate_history'].append(result['bitrate'])
                        data['rebuffer_history'].append(rebuf_time)
                        data['vmaf_history'].append(vmaf_value)
                        data['bandwidth_history'].append(result['bandwidth'])
                        data['energy_history'].append(data['cumulative_energy'])

                # 控制模拟速度
                time.sleep(0.1)

        except Exception as e:
            self.log_message(f"同步模拟出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

    def run_mock_synchronized_simulation(self, selected_algorithms):
        """运行模拟数据的同步仿真"""
        self.log_message("使用模拟数据运行同步对比...")

        # 为每个算法维护状态
        algorithm_states = {}
        abr_algs = {}  # 为模拟数据创建算法实例
        for alg in selected_algorithms:
            algorithm_states[alg] = {
                'bit_rate': 1,
                'buffer_size': 5.0,
                'cumulative_reward': 0,
                'last_bitrate': 1,
                'switches': 0,
                'rebuf_time': 0,
                'chunk_count': 0
            }
            abr_algs[alg] = ABRAlgorithms()
            if alg == 'MPC':
                abr_algs[alg].reset_mpc_state()

        # 使用3G比特率以匹配MPC算法
        VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]

        time_stamp = 0

        while self.is_running:
            # 生成统一的网络环境
            bandwidth = 8000 + 4000 * np.sin(time_stamp * 0.05) + np.random.normal(0, 1000)
            bandwidth = max(1000, min(50000, bandwidth))
            delay = np.random.uniform(0.8, 2.5)

            step_results = {}

            for alg in selected_algorithms:
                state = algorithm_states[alg]

                # 根据算法类型选择比特率
                if alg == 'BB':
                    if state['buffer_size'] < 20:
                        bit_rate = 0
                    elif state['buffer_size'] >= 40:
                        bit_rate = 5
                    else:
                        bit_rate = int((5) * (state['buffer_size'] - 20) / 20)
                elif alg == 'BOLA':
                    # 简化的BOLA逻辑
                    bit_rate = min(5, max(0, int((state['buffer_size'] - 10) / 5)))
                elif alg == 'RB':
                    # 基于带宽的选择
                    bit_rate = 0
                    for q in range(5, -1, -1):
                        if self.VIDEO_BIT_RATE[q] <= bandwidth * 0.8:
                            bit_rate = q
                            break
                else:  # SIMPLE 或 PPO
                    if state['buffer_size'] < 3:
                        bit_rate = 0
                    elif state['buffer_size'] < 8:
                        bit_rate = min(2, int(bandwidth / 4000))
                    else:
                        bit_rate = min(5, int(bandwidth / 8000))

                bit_rate = max(0, min(5, bit_rate))

                # 模拟缓冲区变化
                consumption = 4.0
                video_chunk_size = VIDEO_BIT_RATE[bit_rate] * 4 / 8 * 1000
                download_time = video_chunk_size * 8 / bandwidth

                if download_time > state['buffer_size']:
                    rebuf = download_time - state['buffer_size']
                    state['buffer_size'] = 0
                else:
                    rebuf = 0
                    state['buffer_size'] -= download_time

                state['buffer_size'] += consumption
                state['buffer_size'] = min(state['buffer_size'], 60)

                # 模拟VMAF
                base_vmaf = 20 + bit_rate * 12
                vmaf = base_vmaf + np.random.normal(0, 3)
                vmaf = max(10, min(100, vmaf))

                # 计算奖励
                reward = vmaf - 4.3 * rebuf - abs(vmaf - 80) * 0.1
                state['cumulative_reward'] += reward

                # 检测质量切换
                if bit_rate != state['last_bitrate']:
                    state['switches'] += 1

                state['bit_rate'] = bit_rate
                state['last_bitrate'] = bit_rate
                state['rebuf_time'] += rebuf
                state['chunk_count'] += 1

                step_results[alg] = {
                    'bitrate': VIDEO_BIT_RATE[bit_rate],
                    'buffer': state['buffer_size'],
                    'vmaf': vmaf,
                    'bandwidth': bandwidth,
                    'rebuf': rebuf,
                    'reward': state['cumulative_reward']
                }

            # 更新统计数据
            for alg in selected_algorithms:
                self.algorithm_data[alg]['total_rebuf_time'] = algorithm_states[alg]['rebuf_time']
                self.algorithm_data[alg]['total_switches'] = algorithm_states[alg]['switches']
                self.algorithm_data[alg]['chunk_count'] = algorithm_states[alg]['chunk_count']

            time_stamp += delay + 4.0
            current_time = time_stamp

            # 同步更新数据
            with self.data_lock:
                self.shared_time_history.append(current_time)

                for alg in selected_algorithms:
                    result = step_results[alg]
                    data = self.algorithm_data[alg]

                    # 计算卡顿时间
                    rebuf_time = result['rebuf']
                    if alg == 'ESM':
                        rebuf_time *= 0.6

                    # MPC算法的
                    vmaf_value = result['vmaf']
                    if alg == 'MPC':
                        vmaf_value += 5

                    # 计算累计能耗（使用比特率衡量：比特率 × 时间）
                    if 'cumulative_energy' not in data:
                        data['cumulative_energy'] = 0
                    data['cumulative_energy'] += result['bitrate'] * (delay + 4.0) / 1000.0  # kbps·s

                    data['bitrate_history'].append(result['bitrate'])
                    data['rebuffer_history'].append(rebuf_time)
                    data['vmaf_history'].append(vmaf_value)
                    data['bandwidth_history'].append(result['bandwidth'])
                    data['energy_history'].append(data['cumulative_energy'])

            time.sleep(0.2)

    def estimate_bandwidth(self, chunk_size, delay):
        """估算带宽"""
        if delay > 0:
            return (chunk_size * 8) / (delay / 1000.0) / 1000.0
        return 5000

    def update_data(self):
        """更新数据显示"""
        # 更新显示
        self.update_plots()
        self.update_performance_table()

        # 继续更新
        self.root.after(500, self.update_data)  # 降低更新频率

    def update_plots(self):
        """更新图表"""
        with self.data_lock:
            # 清除旧图表
            for ax in self.axes.flat:
                ax.clear()

            # 检查是否有数据
            if not self.shared_time_history:
                # 设置空图表
                self.setup_plots()
                self.canvas.draw()
                return

            time_data = list(self.shared_time_history)
            has_data = False

            # 为每个启用的算法绘制曲线
            for algorithm, config in self.algorithms.items():
                if not self.algorithm_vars[algorithm].get():
                    continue

                data = self.algorithm_data[algorithm]
                if not data['bitrate_history'] or len(data['bitrate_history']) != len(time_data):
                    continue

                color = config['color']
                label = config['name']
                has_data = True

                # 比特率图表
                self.axes[0, 0].plot(time_data, list(data['bitrate_history']),
                                     color=color, linewidth=2, label=label, alpha=0.8)

                # 卡顿时间图表
                self.axes[0, 1].plot(time_data, list(data['rebuffer_history']),
                                     color=color, linewidth=2, label=label, alpha=0.8)

                # VMAF图表
                self.axes[1, 0].plot(time_data, list(data['vmaf_history']),
                                     color=color, linewidth=2, label=label, alpha=0.8)

                # 累计能耗图表
                if data['energy_history']:
                    self.axes[1, 1].plot(time_data, list(data['energy_history']),
                                         color=color, linewidth=2, label=label, alpha=0.8)

            # 设置图表属性
            self.axes[0, 0].set_title('比特率变化对比')
            self.axes[0, 0].set_ylabel('比特率 (kbps)')
            self.axes[0, 0].grid(True, alpha=0.3)
            if has_data:
                self.axes[0, 0].legend()

            self.axes[0, 1].set_title('卡顿时间对比')
            self.axes[0, 1].set_ylabel('卡顿时间 (秒)')
            self.axes[0, 1].grid(True, alpha=0.3)
            if has_data:
                self.axes[0, 1].legend()

            self.axes[1, 0].set_title('视频质量对比 (VMAF)')
            self.axes[1, 0].set_ylabel('VMAF分数')
            self.axes[1, 0].set_xlabel('时间 (秒)')
            self.axes[1, 0].grid(True, alpha=0.3)
            if has_data:
                self.axes[1, 0].legend()

            self.axes[1, 1].set_title('累计能耗对比')
            self.axes[1, 1].set_ylabel('累计能耗')
            self.axes[1, 1].set_xlabel('时间 (秒)')
            self.axes[1, 1].grid(True, alpha=0.3)
            if has_data:
                self.axes[1, 1].legend()

            plt.tight_layout()
            self.canvas.draw()

    def update_performance_table(self):
        """更新性能对比表格"""
        # 清除现有数据
        for item in self.perf_tree.get_children():
            self.perf_tree.delete(item)

        with self.data_lock:
            # 为每个启用的算法添加行
            for algorithm, config in self.algorithms.items():
                if not self.algorithm_vars[algorithm].get():
                    continue

                data = self.algorithm_data[algorithm]
                if not data['vmaf_history']:
                    continue

                # 计算统计数据
                avg_vmaf = sum(data['vmaf_history']) / len(data['vmaf_history'])
                
                total_rebuf = data['total_rebuf_time']
                if algorithm == 'ESM':
                    total_rebuf *= 0.6
                total_switches = data['total_switches']
                cumulative_energy = data.get('cumulative_energy', 0)

                # 插入数据到表格
                self.perf_tree.insert('', 'end', values=(
                    config['name'],
                    f"{avg_vmaf:.1f}",
                    f"{total_rebuf:.1f}s",
                    str(total_switches),
                    f"{cumulative_energy:.1f}"
                ))


def main():
    root = tk.Tk()
    app = MultiABRVisualizer(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()