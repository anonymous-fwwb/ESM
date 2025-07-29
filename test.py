import os
import sys
import numpy as np
import trace_loader
import ppo_h as network
import static_env as env

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

S_INFO = 8
S_LEN = 8
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
VIDEO_BIT_RATE = [1000, 2500, 5000, 8000, 16000, 40000]  # 5G
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log_sim_esm'
TEST_TRACES = './test/'
# NN_MODEL = sys.argv[1]
NN_MODEL = './models/nn_model_ep_71400.pth'


def main():
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM
    all_cooked_time, all_cooked_bw, all_file_names = trace_loader.load_trace(TEST_TRACES)
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')
    actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                            learning_rate=ACTOR_LR_RATE)

    if NN_MODEL is not None:
        actor.load_model(NN_MODEL)

    time_stamp = 0

    bit_rate = DEFAULT_QUALITY
    last_vmaf = None

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    while True:
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            next_video_chunk_vmaf, vmaf, \
            end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        if last_vmaf is None:
            last_vmaf = vmaf

        reward = 0.8469011 * vmaf - 28.79591348 * rebuf + 0.29797156 * \
                 np.abs(np.maximum(vmaf - last_vmaf, 0.)) - 1.06099887 * \
                 np.abs(np.minimum(vmaf - last_vmaf, 0.)) - \
                 2.661618558192494

        r_batch.append(reward)

        last_vmaf = vmaf

        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(vmaf) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(entropy_) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        state = np.roll(state, -1, axis=1)

        state[0, -1] = vmaf / 100.
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        state[4] = -1.
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        state[5] = -1.
        state[5, :A_DIM] = np.array(next_video_chunk_vmaf) / 100.
        state[6, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        state[7, :2] = np.array([0.34, 0.21])

        # PPO-H
        action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        noise = np.random.gumbel(size=len(action_prob))
        # bit_rate = np.argmax(np.log(action_prob) + noise)
        bit_rate = np.argmax(np.log(action_prob))
        if buffer_size >= 35:
            bit_rate = 5

        s_batch.append(state)
        entropy_ = -np.dot(action_prob, np.log(action_prob))
        entropy_record.append(entropy_)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            bit_rate = DEFAULT_QUALITY

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
