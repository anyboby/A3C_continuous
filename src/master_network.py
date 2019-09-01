import tensorflow as tf
import numpy as np
import constants as Constants
import network_shares as Netshare
class MasterNetwork(object):
    def __init__(self, scope, globalAC=None):

        if scope == Constants.GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                #print ("scope: " + str(scope))
                self.s = tf.placeholder(tf.float32, Netshare.DIM_S, 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                #print ("scope: " + str(scope))
                self.s = tf.placeholder(tf.float32, Netshare.DIM_S, 'S')
                self.a_his = tf.placeholder(tf.float32, Netshare.DIM_A, 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))



                #choose actions from normal dist
                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * Netshare.BOUND_A[1], sigma + 1e-4
                normal_dist = tf.distributions.Normal(mu, sigma)
                
                #print("mu shape: " + str(mu.shape))
                #print("sigma shape: " + str(sigma.shape))
                #print("normal shape: " + str(normal_dist))

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = Constants.ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), Netshare.BOUND_A[0], Netshare.BOUND_A[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                #print("sigma shape: " + str(self.A))


            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = Netshare.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = Netshare.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            ######## CarRacing Actor ######### 
            l_conv1 = tf.layers.conv2d(self.s, 16, (8,8), strides=(3,3), activation=tf.nn.relu6, kernel_initializer=w_init, name="conv1")
            l_conv2 = tf.layers.conv2d(l_conv1, 8, (4,4), strides=(2,2), activation=tf.nn.relu6, kernel_initializer=w_init, name="conv2")
            l_fl = tf.layers.flatten(l_conv2, name='fl_a')
            l_a = tf.layers.dense(l_fl, 250, tf.nn.relu6, kernel_initializer=w_init, name='la')
            # N_A[0] has None placeholder for samples, so the real number of actions if in N_A[1]
            mu = tf.layers.dense(l_a, Netshare.DIM_A[1], tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, Netshare.DIM_A[1], tf.nn.softplus, kernel_initializer=w_init, name='sigma')
            ####################################
            
            
            ######## Pendulum Actor ######### 
            # l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            # mu = tf.layers.dense(l_a, Netshare.DIM_A[1], tf.nn.tanh, kernel_initializer=w_init, name='mu')
            # sigma = tf.layers.dense(l_a, Netshare.DIM_A[1], tf.nn.softplus, kernel_initializer=w_init, name='sigma')
            ####################################
            
        with tf.variable_scope('critic'):
            
            ######## CarRacing Critic ########
            l_c = tf.layers.dense(l_fl, 125, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
            ##################################


            ######## Pendulum Critic ########
            # l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            # v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
            ##################################

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        Netshare.SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        Netshare.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        result = Netshare.SESS.run(self.A, {self.s: s})
        return result