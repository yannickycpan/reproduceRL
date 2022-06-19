import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import numpy as np
import tensorflow as tf
from FunctionApproximator import FunctionApproximator
from network.actor_critic_network import create_actor_nn, create_critic_nn
from network.operations import update_target_nn_assign, update_target_nn_move
from Agent import Agent


class DDPG(FunctionApproximator):
    def __init__(self, params):
        # NOTE the first assignment
        super(DDPG, self).__init__(params)
        # use the same setting as used in the paper
        self.actor_lc = params['alpha']
        self.critic_lc = params['critic_factor'] * params['alpha']

        self.create_critic = create_critic_nn

        if 'NoTarget' in self.agent_name:
            self.computeQtargets = self.computeQtargets_wotar
        else:
            self.computeQtargets = self.computeQtargets_wtar

        with self.g.as_default():
            # create critic NN and target critic NN
            self.actor_input, self.actor_output, self.actor_vars \
                = create_actor_nn('actor', self.stateDim, self.actionDim, self.actionBound, self.n_h1, self.n_h2,
                                  self.usetanh)
            self.target_actor_input, self.target_actor_output, self.target_actor_vars \
                = create_actor_nn('target_actor', self.stateDim, self.actionDim, self.actionBound, self.n_h1,
                                  self.n_h2, self.usetanh)
            self.critic_input_s, self.critic_input_a, self.critic_output, self.critic_vars \
                = self.create_critic('critic', self.stateDim, 1, self.actor_output, self.n_h1, self.n_h2, self.usetanh)
            self.target_critic_input_s, self.target_critic_input_a, self.target_critic_output, self.target_critic_vars \
            = self.create_critic('target_critic', self.stateDim, 1, self.target_actor_output, self.n_h1, self.n_h2, self.usetanh)
            # create ops to update critic and actor
            self.critic_value_holders, self.critic_update = self.update_critic_ops(self.critic_lc)
            self.actor_update = self.update_actor_ops(self.actor_lc)
            # init target NN the same variable values
            self.tar_tvars = self.target_actor_vars + self.target_critic_vars
            self.tvars = self.actor_vars + self.critic_vars

            self.target_params_init = update_target_nn_assign(self.tar_tvars, self.tvars)
            self.target_params_update = update_target_nn_move(self.tar_tvars, self.tvars, self.tau)

            self.gradvars = tf.gradients(self.critic_output, self.critic_vars)[0]

            # init session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.target_params_init)
            '''init saver'''
            self.saver = tf.train.Saver()

    def update_actor_ops(self, lr):
        with self.g.as_default():
            critic_output_mean = -tf.reduce_mean(self.critic_output)
            actor_grads = tf.gradients(critic_output_mean, self.actor_vars)
            actor_grads_feed = zip(actor_grads, self.actor_vars)
            actor_update = tf.train.AdamOptimizer(lr, name='actor_opt').apply_gradients(actor_grads_feed)
        return actor_update

    def update_critic_ops(self, lr):
        with self.g.as_default():
            critic_value_holders = tf.placeholder(tf.float32, [None], name='q_holders')
            critic_loss = tf.reduce_mean(tf.square(critic_value_holders - tf.squeeze(self.critic_output)))
            critic_update_grads = tf.gradients(critic_loss, self.critic_vars)
            critic_grads_feed = zip(critic_update_grads, self.critic_vars)
            critic_update = tf.train.AdamOptimizer(lr, name='critic_opt').apply_gradients(critic_grads_feed)
        return critic_value_holders, critic_update

    '''return an action to take for each state, NOTE this action is in [0, 1]'''
    def take_action(self, state):
        # print state.shape
        action = self.sess.run(self.actor_output,
                               {self.actor_input: np.expand_dims(state, 0)})
        return action[0]

    def computeQtargets_wtar(self, state_tp, reward, gamma):
        qvalues = self.sess.run(self.target_critic_output,
                                {self.target_critic_input_s: state_tp, self.target_actor_input: state_tp})
        qtargets = reward + gamma * np.squeeze(qvalues)
        return qtargets

    def computeQtargets_wotar(self, state_tp, reward, gamma):
        qvalues = self.sess.run(self.critic_output,
                                {self.critic_input_s: state_tp, self.actor_input: state_tp})
        qtargets = reward + gamma * np.squeeze(qvalues)
        return qtargets

    def train(self, state, action, state_tp, reward, gamma):
        qtargets = self.computeQtargets(state_tp, reward/self.reward_scale, gamma)
        if len(action.shape) < 2:
            action = action.reshape((-1, self.actionDim))
        self.sess.run(self.critic_update, feed_dict={self.critic_input_s: state, self.critic_input_a: action,
                                                     self.critic_value_holders: qtargets})
        self.sess.run(self.actor_update,
                      feed_dict={self.actor_input: state, self.critic_input_s: state})
        if 'NoTarget' not in self.agent_name:
            self.sess.run(self.target_params_update)
        return None


class DDPGAgent(Agent):
    def __init__(self, params):
        super(DDPGAgent, self).__init__(params)
        self.max_reward = 0.0
        self.n_episode = 0.
        self.noise_t = np.zeros(self.actionDim)

    def take_action(self, state):
        if not self.start_learning:
            return np.random.uniform(-self.actionBound, self.actionBound, self.actionDim)
        action = self.agent_function.take_action(state)
        self.noise_t += np.random.normal(0, 0.2, self.actionDim) - self.noise_t * 0.15
        action = action + self.noise_t
        return np.clip(action, -self.actionBound, self.actionBound)

    '''here we use and store option, not primal actions, so primal action is not directly used for training'''
    '''they passed a in this function is the primal action, not option'''
    def update(self, s, a, sp, r, episodeEnd, info):
        gamma = self.gamma if not episodeEnd else 0.0
        if episodeEnd:
            self.n_episode += 1.
        self.replaybuffer.add(s, a, sp, r, gamma)
        self.n_samples += 1
        if self.replaybuffer.getSize() >= self.warm_up_steps:
            self.start_learning = True
            for _ in range(self.planningSteps):
                bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.batchSize)
                self.agent_function.train(bs, ba, bsp, br, bgamma)