import os
import numpy as np
import dynamic_env
import trace_loader

S_INFO = 8
S_LEN = 8
A_DIM = 6
TRAIN_SEQ_LEN = 100
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
DEFAULT_QUALITY = 1
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6


def calculate_energy(alpha, beta, th, data_sizes):
    th_inverse = 1 / th

    energy = (alpha * th_inverse + beta) * data_sizes
    return energy


class ABREnv():
    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        all_cooked_time, all_cooked_bw, _ = trace_loader.load_trace()
        self.net_env = dynamic_env.Environment(all_cooked_time=all_cooked_time,
                                               all_cooked_bw=all_cooked_bw,
                                               random_seed=random_seed)

        self.last_bit_rate = DEFAULT_QUALITY
        self.last_vmaf = 0.
        self.beta, self.gamma, self.eta = np.random.uniform(), np.random.uniform(), np.random.uniform()

        self.buffer_size = 0.
        self.state = np.zeros((S_INFO, S_LEN))

    def seed(self, num):
        np.random.seed(num)

    def reset(self):
        self.time_stamp = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_size = 0.
        self.beta, self.gamma, self.eta = np.random.uniform(), np.random.uniform(), np.random.uniform()

        self.state = np.zeros((S_INFO, S_LEN))
        bit_rate = self.last_bit_rate
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            next_video_chunk_vmaf, vmaf, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate)
        state = np.roll(self.state, -1, axis=1)

        self.last_vmaf = vmaf

        state[0, -1] = vmaf / 100.

        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        state[4] = -1.
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        state[5] = -1.
        state[5, :A_DIM] = np.array(next_video_chunk_vmaf) / 100.
        state[6, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        state[7, :3] = np.array([self.beta, self.gamma, self.eta])

        self.state = state
        return state

    def render(self):
        return

    def step(self, action):
        bit_rate = int(action)

        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            next_video_chunk_vmaf, vmaf, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        alpha = 210  # α 的值
        beta = 28  # β 的值
        th = float(video_chunk_size) / float(delay) / M_IN_K

        # 计算能耗
        energy = calculate_energy(alpha, beta, th, video_chunk_size) + 24.71 * VIDEO_BIT_RATE[action] + 1121.5

        # 拆分公式的各部分
        vmaf_reward = vmaf
        rebuf_penalty = self.beta * 100. * rebuf
        vmaf_difference_penalty = self.gamma * 5. * np.abs(vmaf - self.last_vmaf)
        energy_penalty = self.eta * 1e-8 * energy

        # 总的 reward 计算
        reward = vmaf_reward - rebuf_penalty - vmaf_difference_penalty - energy_penalty

        self.last_bit_rate = bit_rate
        self.last_vmaf = vmaf

        state = np.roll(self.state, -1, axis=1)

        state[0, -1] = vmaf / 100.
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4] = -1.
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5] = -1.
        state[5, :A_DIM] = np.array(next_video_chunk_vmaf) / 100.
        state[6, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        state[7, :3] = np.array([self.beta, self.gamma, self.eta])

        self.state = state
        return state, reward, end_of_video, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf}
