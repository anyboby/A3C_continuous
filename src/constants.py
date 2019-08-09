import numpy as np
import tensorflow as tf
import multiprocessing
import threading

#Pendulum-v0
#CarRacing-v0
GAME = 'CarRacing-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = 4   #multiprocessing.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 2000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
TF_DEVICE = "/cpu:0"
