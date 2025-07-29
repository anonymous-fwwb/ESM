import multiprocessing as mp
import numpy as np
import os
from ABRenv import ABREnv
import ppo_h as network
import torch
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

S_DIM = [8, 8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 1000
TRAIN_EPOCH = 500000
MODEL_SAVE_INTERVAL = 300
RANDOM_SEED = 42
SUMMARY_DIR = './models'
TRAIN_TRACES = './train/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = SUMMARY_DIR + '/log'

if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None


def testing(epoch, nn_model, log_file):
    if os.path.exists(TEST_LOG_FOLDER):
        shutil.rmtree(TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)

    os.system('python test.py ' + nn_model)

    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        entropies.append(np.mean(entropy[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies)


def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    with open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        actor = network.Network(state_dim=S_DIM,
                                action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        nn_model = NN_MODEL
        if nn_model is not None:
            actor.load_model(nn_model)
            print('Model Loaded.')

        for epoch in range(TRAIN_EPOCH):
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, r = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, r_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                r += r_
            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(r)

            actor.train(s_batch, a_batch, p_batch, v_batch, epoch)

            if epoch % MODEL_SAVE_INTERVAL == 0:
                actor.save_model(SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth')
                testing(epoch, SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth', test_log_file)


def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id)
    actor = network.Network(state_dim=S_DIM, action_dim=A_DIM,
                            learning_rate=ACTOR_LR_RATE)

    actor_net_params = net_params_queue.get()
    actor.set_network_params(actor_net_params)

    for epoch in range(TRAIN_EPOCH):
        obs = env.reset()
        s_batch, a_batch, p_batch, r_batch = [], [], [], []
        for step in range(TRAIN_SEQ_LEN):
            s_batch.append(obs)

            action_prob = actor.predict(
                np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

            noise = np.random.gumbel(size=len(action_prob))
            bit_rate = np.argmax(np.log(action_prob) + noise)

            obs, rew, done, info = env.step(bit_rate)

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            a_batch.append(action_vec)
            r_batch.append(rew)
            p_batch.append(action_prob)
            if done:
                break
        v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
        exp_queue.put([s_batch, a_batch, p_batch, v_batch])

        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)


def main():
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    coordinator.join()


if __name__ == '__main__':
    main()
