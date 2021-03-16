import tensorflow as tf
import numpy as np
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from FunctionApproximator import FunctionApproximator
from network.dqn_network import create_qnn_inputsparse
from network.operations import compute_q_acted, update_target_nn_assign, update_critic_ops
from Agent import Agent
from network.ftann import FTA


class DQN(FunctionApproximator):
    def __init__(self, params):
        super(DQN, self).__init__(params)

        self.create_qnn = create_qnn_inputsparse
        self.unittype = tf.nn.tanh if self.usetanh == 1 else tf.nn.relu
        self.statescale = 255. if self.use_atari_nn else 1.0

        if 'NoTarget' in self.agent_name:
            self.update_target_frequency = 1
            self.compute_Qtarget = self.compute_Qtarget_wotar
        else:
            self.compute_Qtarget = self.compute_Qtarget_wtar

        print('statedim, actiondim, tiles, use atari, target nn move rate are ---------------------------------- ',
              self.stateDim, self.actionDim, self.create_qnn, self.update_target_frequency,
              self.use_atari_nn, self.statescale)

        with self.g.as_default():
            self.fta = FTA(params)
            '''used for batch normalization'''
            self.action_input = tf.placeholder('int64', [None])
            self.state_input, self.q_values, self.max_q_value, self.best_act, self.phi, self.tvars = \
                self.create_qnn(self.agent_name, self.dtype, self.stateDim, self.actionDim, self.n_h1, self.n_h2,
                                self.unittype, self.fta)
            ''' for target network a new FTA needs to be created '''
            self.tar_state_input, self.tar_q_values, self.tar_max_q_value, self.tar_best_act, self.tar_phi, self.tar_tvars \
                = self.create_qnn("target_"+self.agent_name, self.dtype, self.stateDim, self.actionDim, self.n_h1, self.n_h2,
                                  self.unittype, FTA(params))
            print('q value shape is ----------------- ', self.q_values.shape)
            # define state action value
            self.sa_value, self.tar_sa_value \
                = compute_q_acted("sa_values", self.action_input, self.actionDim, self.q_values, self.tar_q_values)
            # define loss operation, weight has been incorporated in fta_loss
            self.qtarget_input, self.params_update = update_critic_ops(self.sa_value, self.learning_rate,
                                                                           self.fta.fta_loss, 1.0)
            # define optimization
            self.target_params_update = update_target_nn_assign(self.tar_tvars, self.tvars)
            self.gradvars = tf.gradients(self.max_q_value, self.tvars)[0]
            # initialize network
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.target_params_update)
            self.saver = tf.train.Saver()

            self.sparse_dim = int(self.phi.shape[1])
            print(' the sparse dim is ===================================== ', self.sparse_dim)

    def compute_Qtarget_wtar(self, sp, r, gamma):
        max_q = self.sess.run(self.tar_max_q_value, feed_dict={self.tar_state_input: sp})
        qtarget = r + gamma * max_q
        return qtarget

    def compute_Qtarget_wotar(self, sp, r, gamma):
        max_q = self.sess.run(self.max_q_value, feed_dict={self.state_input: sp})
        qtarget = r + gamma * max_q
        return qtarget

    def take_action(self, state):
        if len(state.shape) < 2:
            state = state[None, :]
        act = self.sess.run(self.best_act, feed_dict={self.state_input: state/self.statescale})
        return act[0]

    def train(self, s, a, sp, r, gamma):
        qtarget = self.compute_Qtarget(sp/self.statescale, r/self.reward_scale, gamma)
        self.sess.run(self.params_update,
                      feed_dict={self.state_input: s/self.statescale,
                                 self.qtarget_input: qtarget, self.action_input: a})
        if self.update_count % self.update_target_frequency == 0 and 'NoTarget' not in self.agent_name:
            self.sess.run(self.target_params_update)
        self.update_count += 1


''' this implementation is a modification of 
    Two Geometric Input Transformation Methods for 
    Fast Online Reinforcement Learning with Neural Nets '''


class TCNNAgent(Agent):
    def __init__(self, params):
        super(TCNNAgent, self).__init__(params)
        self.agent_function = DQN(params)

    def take_action(self, state):
        if np.random.uniform(0.0, 1.0) < self.epsilon or not self.start_learning:
            action = np.random.randint(self.actionDim)
        else:
            action = self.agent_function.take_action(state)
        return action

    def update(self, s, a, sp, r, episodeEnd, info):
        gamma = self.gamma if not episodeEnd else 0.0
        ''' epsilon is decayed only in atari games '''
        self.linear_decay_epsilon()
        self.replaybuffer.add(s, a, sp, r, gamma)
        self.n_samples += 1
        if self.replaybuffer.getSize() >= self.warm_up_steps and self.n_samples % self.trainFrequency == 0:
            self.start_learning = True
            for pn in range(self.planningSteps):
                bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.batchSize)
                self.agent_function.train(bs, ba, bsp, br, bgamma)