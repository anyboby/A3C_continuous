import tensorflow as tf
import threading

OPT_A = None
OPT_C = None

with threading.Lock():
    COORD = tf.train.Coordinator()
SESS = None

""" examples:
N_S = [84,84,4]
N_A = [3]
A_BOUND = [-1,1]
"""

N_S = [84,84,4]
N_A = [3]
A_BOUND = [-1,1]