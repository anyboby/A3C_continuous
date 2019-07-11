"""
inspired by morvanzhou
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

from master_network import MasterNetwork
from agent import Agent
import constants as Constants
import network_shares as Netshare

if __name__ == "__main__":
    Netshare.SESS = tf.Session()
    env = gym.make(Constants.GAME)

    Netshare.N_S = env.observation_space.shape[0]
    Netshare.N_A = env.action_space.shape[0]
    Netshare.A_BOUND = [env.action_space.low, env.action_space.high]

    print ("N_S: " + str(Netshare.N_S))
    print ("N_A: " + str(Netshare.N_A))
    print ("A_BOUND: " + str(Netshare.A_BOUND))

    with tf.device("/cpu:0"):
        #TODO die optimizer aufräumen
        Netshare.OPT_A = tf.train.RMSPropOptimizer(Constants.LR_A, name='RMSPropA')
        Netshare.OPT_C = tf.train.RMSPropOptimizer(Constants.LR_C, name='RMSPropC')
        GLOBAL_AC = MasterNetwork(Constants.GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(Constants.N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Agent(i_name, GLOBAL_AC))

    Netshare.SESS.run(tf.global_variables_initializer())

    if Constants.OUTPUT_GRAPH:
        if os.path.exists(Constants.LOG_DIR):
            shutil.rmtree(Constants.LOG_DIR)
        tf.summary.FileWriter(Constants.LOG_DIR, Netshare.SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    Netshare.COORD.join(worker_threads)

    plt.plot(np.arange(len(Constants.GLOBAL_RUNNING_R)), Constants.GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
