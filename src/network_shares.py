import tensorflow as tf


OPT_A = None
OPT_C = None
COORD = tf.train.Coordinator()
SESS = None


N_S = [84,84,4]
N_A = [3]
A_BOUND = [-1,1]