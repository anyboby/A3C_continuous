import numpy as np
import tensorflow as tf
import multiprocessing
import threading

#Pendulum-v0
#CarRacing-v0
GAME = "CarRacing-v0"
OUTPUT_GRAPH = True
LOG_DIR = "./log"
N_WORKERS = 4   #multiprocessing.cpu_count()
MAX_EP_STEP = 1000
MAX_GLOBAL_EP = 3000
GLOBAL_NET_SCOPE = "Global_Net"
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
TF_DEVICE = "/cpu:0"
EARLY_TERMINATION = 15 # score difference between epMax and current score for termination

# network constants
# manual_dims is activated, state and action space can be manually set
# if deactivated, state and action space of env are used automatically as 
# network in/output
manual_dims = True
STATE_STACK = 4
STATE_WIDTH = 84
STATE_HEIGHT = 84
DIMS_S = [STATE_WIDTH, STATE_HEIGHT, STATE_STACK]
ACTIONS = 3
DIMS_A = [ACTIONS]

#OPENCV
WAITKEY = 1
