import numpy as np
import pandas as pd
import tensorflow as tf
import tf_utils
import sys
import model_exp1 as mdl

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.5,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            is_train=False,
            weight_folder="",
            episode = 1000,
            is_monitor = False
    ):
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.is_train = is_train
        self.is_monitor = is_monitor
        self.weight_folder = weight_folder
        self.episode = episode

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory_trans = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_ID = np.zeros((self.memory_size,7,256,256,3*2))
        self.memory_PO = np.zeros((self.memory_size,21*2))
        
        
        # consist of [target_net, evaluate_net]
        self._build_net()
        
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        
        with tf.variable_scope('hard_replacement'):
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        
        self.print_trainable_parameters(self.e_params)
        
        # Create a saver
        self.saver = tf.train.Saver(tf.global_variables())
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        if not self.is_train:
            self.load_weight(self.weight_folder)
            print("loading weights")
        else:
            self.sess.run(tf.global_variables_initializer())
            print("initial weights")
            
        self.cost_his = []
        
    def print_trainable_parameters(self, net):
        total_parameters = 0
        for variable in net:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()

            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value

            total_parameters += variable_parameters
            
        print('total_parameters', total_parameters)
        
 
    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.I = tf.placeholder(tf.float32, [None, 7, 256, 256, 3], name='I')
        self.D = tf.placeholder(tf.float32, [None, 7, 256, 256, 3], name='D')
        self.P = tf.placeholder(tf.float32, [None, 21], name='P')
        self.O = tf.placeholder(tf.float32, [None, 21], name='O')
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        
        self.train_phase = tf.placeholder(tf.bool, name='phase_train')
        
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        
        with tf.variable_scope('eval_net'):
            self.q_eval = mdl.model(self.s,self.I,self.D,self.P,self.O,self.train_phase,self.n_actions)
        
        with tf.variable_scope('target_net'):
            self.q_next = mdl.model(self.s_,self.I,self.D,self.P,self.O,self.train_phase,self.n_actions)
        
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
            
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
            
        with tf.variable_scope('loss'):
            l2_loss = tf.losses.get_regularization_loss()
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            self.loss = self.loss + l2_loss
            
        with tf.variable_scope('train'):
#             self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    
            
    def store_transition(self, s, a, r, s_, env):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # print("s:{0}, a:{1}, r:{2}, s_:{3}".format(s, a, r, s_))
        transition = np.hstack((s, [a, r], s_))
        if self.is_monitor:
            print('Store:', transition)
        image_depth = np.concatenate((env[0], env[1]),axis = 3)
        position_orientation = np.hstack(((env[4]-env[2]).flatten(), (env[5]-env[3]).flatten()))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory_trans[index, :] = transition
        self.memory_PO[index, :] = position_orientation
        self.memory_ID[index, :] = image_depth
        self.memory_counter += 1

    def choose_action(self, observation, step):
        # to have batch dimension when feed into tf placeholder
        if step > 100000:
            self.epsilon = 0.7  
        elif step > 200000:
            self.epsilon = 0.9
            
        if np.random.uniform() < self.epsilon:
            # print('select action')
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation[0][np.newaxis, :],
                                                                  self.I: observation[1][np.newaxis, :]/255,
                                                                  self.D: observation[2][np.newaxis, :]/100,
                                                                  self.P: (observation[5]-observation[3]).flatten()[np.newaxis, :]*10,
                                                                  self.O: (observation[6]-observation[4]).flatten()[np.newaxis, :]/180,
                                                                  self.train_phase: False})
            action = np.argmax(actions_value)
            if not self.is_train or self.is_monitor:
                print('[Greedy]',action)
        else:
            action = np.random.randint(0, self.n_actions-np.max(observation[0]))
            if self.is_monitor:
                print('[Random(%d,%d)] %d' % (0, self.n_actions-np.max(observation[0]),action))
        return action

    def learn(self):
        is_update = False
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
#             print('\ntarget_params_replaced\n')
            is_update = True

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            
        batch_memory_trans = self.memory_trans[sample_index, :]
        batch_memory_ID = self.memory_ID[sample_index, :]
        batch_memory_PO = self.memory_PO[sample_index, :]
        
#         print( batch_memory_ID[:, :, :, :3].shape)
#         print( batch_memory_ID[:, :, :, 3:].shape)
#         print(batch_memory_PO[:,:21].shape)
#         print(batch_memory_PO[:,21:].shape)

        _, cost, _ = self.sess.run([self._train_op, self.loss, self.update_ops],
                                     feed_dict={self.s: batch_memory_trans[:, :self.n_features],
                                                self.I: batch_memory_ID[:, :, :, :, :3]/255,
                                                self.D: batch_memory_ID[:, :, :, :, 3:]/100,
                                                self.P: batch_memory_PO[:,:21]*10,
                                                self.O: batch_memory_PO[:,21:]/180,
                                                self.s_: batch_memory_trans[:, -self.n_features:],  # fixed params
                                                self.r: batch_memory_trans[:, self.n_features + 1],
                                                self.a: batch_memory_trans[:, self.n_features],  # fixed params
                                                self.train_phase: True})

        self.cost_his.append(cost)
#         print(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return is_update

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylim(ymax = 50, ymin = 0)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
        
    def save_weight(self, episode, path):
        self.saver.save(self.sess, path+str(episode), self.global_step)
        
    def load_weight(self,path):
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("------------load fail--------------")
            sys.exit()