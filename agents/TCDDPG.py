import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import numpy as np
import tensorflow as tf
from FunctionApproximator import FunctionApproximator
from network.actor_critic_network import create_actor_nn, create_actor_fta_nn, create_critic_fta_nn_vanilla
from network.operations import update_target_nn_assign, update_target_nn_move
from Agent import Agent
from sparseutils import tfget_sparsities_state_action_pair
from network.ftann import FTA


class DDPG(FunctionApproximator):
    def __init__(self, params):
        # NOTE the first assignment
        super(DDPG, self).__init__(params)
        # use the same setting as used in the paper
        self.actor_lc = params['alpha']
        self.critic_lc = params['critic_factor'] * params['alpha']

        self.sparseactor = params['sparseactor']

        self.sample_count = 0

        self.create_critic = create_critic_fta_nn_vanilla
        self.create_actor = create_actor_fta_nn

        if 'NoTarget' in self.agent_name:
            self.computeQtargets = self.computeQtargets_wotar
        elif 'ActorTarget' in self.agent_name:
            self.computeQtargets = self.computeQtargets_wactortar
        else:
            self.computeQtargets = self.computeQtargets_wtar

        with self.g.as_default():
            self.actor_fta = FTA(params)
            self.critic_fta = FTA(params)
            ''' create actor network '''
            if self.sparseactor == 1:
                self.actor_input, self.actor_output, self.actorhiddenphi, self.actor_vars \
                    = self.create_actor('actor',  self.stateDim, self.actionDim, self.actionBound,
                                        self.n_h1, self.n_h2, self.usetanh, self.actor_fta)
                self.target_actor_input, self.target_actor_output, _, self.target_actor_vars \
                    = self.create_actor('target_actor',  self.stateDim, self.actionDim, self.actionBound,
                                        self.n_h1, self.n_h2, self.usetanh, self.actor_fta)
                print(' sparse actor also used ======================================================== ')
            else:
                self.actor_input, self.actor_output, self.actor_vars \
                    = create_actor_nn('actor', self.stateDim, self.actionDim, self.actionBound, self.n_h1, self.n_h2,
                                      self.usetanh)
                self.target_actor_input, self.target_actor_output, self.target_actor_vars \
                    = create_actor_nn('target_actor', self.stateDim, self.actionDim, self.actionBound, self.n_h1,
                                      self.n_h2, self.usetanh)
            ''' create critic network '''
            self.critic_input_s, self.critic_input_a, self.critic_output, self.phi, self.critic_vars \
                = self.create_critic(self.agent_name, self.stateDim, 1, self.actor_output,
                                     self.n_h1, self.n_h2, self.usetanh, self.critic_fta)
            self.target_critic_input_s, self.target_critic_input_a, self.target_critic_output,\
                self.tar_phi, self.target_critic_vars \
                = self.create_critic('target_'+self.agent_name, self.stateDim, 1, self.target_actor_output,
                                     self.n_h1, self.n_h2, self.usetanh, self.critic_fta)
            # create ops to update critic and actor
            self.critic_value_holders, self.critic_update = self.update_critic_ops(self.critic_lc)
            self.actor_update = self.update_actor_ops(self.actor_lc)
            # init target NN the same variable values
            self.tar_tvars = self.target_actor_vars + self.target_critic_vars
            self.tvars = self.actor_vars + self.critic_vars

            self.gradvars = tf.gradients(self.critic_output, self.critic_vars)[0]
            self.target_params_init = update_target_nn_assign(self.tar_tvars, self.tvars)
            self.target_params_update = update_target_nn_move(self.tar_tvars, self.tvars, self.tau)
            self.target_actor_params_update = update_target_nn_move(self.target_actor_vars, self.actor_vars, self.tau)

            # init session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.target_params_init)
            ''' init saver '''
            self.saver = tf.train.Saver()

            ''' sparse dim '''
            self.sparse_dim = int(self.phi.shape[1])
            self.state_visit_count = np.zeros(self.sparse_dim)

    def update_actor_ops(self, lr):
        with self.g.as_default():
            critic_output_mean = -tf.reduce_mean(self.critic_output) \
                                 + self.actor_fta.fta_loss
            actor_grads = tf.gradients(critic_output_mean, self.actor_vars)
            actor_grads_feed = zip(actor_grads, self.actor_vars)
            actor_update = tf.train.AdamOptimizer(lr, name='actor_opt').apply_gradients(actor_grads_feed)
        return actor_update

    def update_critic_ops(self, lr):
        with self.g.as_default():
            critic_value_holders = tf.placeholder(tf.float32, [None], name='q_holders')
            critic_loss = tf.reduce_mean(tf.square(critic_value_holders - tf.squeeze(self.critic_output))) \
                          + self.critic_fta.fta_loss
            critic_update_grads = tf.gradients(critic_loss, self.critic_vars)
            critic_grads_feed = zip(critic_update_grads, self.critic_vars)
            critic_update = tf.train.AdamOptimizer(lr, name='critic_opt').apply_gradients(critic_grads_feed)
        return critic_value_holders, critic_update

    '''return an action to take for each state, NOTE this action is in [0, 1]'''
    def take_action(self, state):
        action = self.sess.run(self.actor_output,
                               {self.actor_input: np.expand_dims(state, 0)})
        return action[0]

    def computeQtargets_wtar(self, state_tp, reward, gamma):
        qvalues = self.sess.run(self.target_critic_output,
                                {self.target_critic_input_s: state_tp, self.target_actor_input: state_tp})
        qtargets = reward + gamma * np.squeeze(qvalues)
        return qtargets

    def computeQtargets_wactortar(self, state_tp, reward, gamma):
        actions = self.sess.run(self.target_actor_output, {self.target_actor_input: state_tp})
        qvalues = self.sess.run(self.critic_output,
                                {self.critic_input_s: state_tp, self.critic_input_a: actions})
        qtargets = reward + gamma * np.squeeze(qvalues)
        return qtargets

    def computeQtargets_wotar(self, state_tp, reward, gamma):
        qvalues = self.sess.run(self.critic_output,
                                {self.critic_input_s: state_tp, self.actor_input: state_tp})
        qtargets = reward + gamma * np.squeeze(qvalues)
        return qtargets

    def update_target_nn(self):
        if 'ActorTarget' in self.agent_name:
            self.sess.run(self.target_actor_params_update)
        elif 'NoTarget' not in self.agent_name:
            self.sess.run(self.target_params_update)
        self.update_count += 1

    def train(self, state, action, state_tp, reward, gamma):
        qtargets = self.computeQtargets(state_tp, reward/self.reward_scale, gamma)
        self.sess.run(self.critic_update, feed_dict={self.critic_input_s: state, self.critic_input_a: action,
                                                     self.critic_value_holders: qtargets})
        self.sess.run(self.actor_update,
                      feed_dict={self.actor_input: state, self.critic_input_s: state})
        self.update_target_nn()


class TCDDPGAgent(Agent):
    def __init__(self, params):
        super(TCDDPGAgent, self).__init__(params)
        self.agent_function = DDPG(params)

        self.max_reward = 0.0

        self.n_episode = 0.
        self.noise_t = np.zeros(self.actionDim)

        self.notTrain = False


    def take_action(self, state):
        if not self.start_learning:
            return np.random.uniform(-self.actionBound, self.actionBound, self.actionDim)
        action = self.agent_function.take_action(state)
        self.noise_t += np.random.normal(0, 0.2, self.actionDim) - self.noise_t * 0.15
        action = action + self.noise_t * self.conti_noisescale
        return np.clip(action, -self.actionBound, self.actionBound)

    def custermized_log(self, logger):
        if 'OverlapSparse' not in logger.logger_dict:
            logger.logger_dict['OverlapSparse'] = []
        if 'InstanceSparse' not in logger.logger_dict:
            logger.logger_dict['InstanceSparse'] = []
        if self.replaybuffer.getSize() >= self.batchSize:
            overlapsp, instancesp = tfget_sparsities_state_action_pair(self.agent_function.sess,
                                                                       self.agent_function.critic_input_s,
                                                                       self.agent_function.critic_input_a,
                                                     self.agent_function.phi, self.replaybuffer,
                                                     self.agent_function.batchSize, 1.0)
            logger.logger_dict['OverlapSparse'].append(overlapsp)
            logger.logger_dict['InstanceSparse'].append(instancesp)

            print(' the overlap, instance, critic, actor bounds are :: ========================================== ',
                  overlapsp, instancesp)


    '''here we use and store option, not primal actions, so primal action is not directly used for training'''
    '''they passed a in this function is the primal action, not option'''
    def update(self, s, a, sp, r, episodeEnd, info):
        gamma = self.gamma if not episodeEnd else 0.0
        if episodeEnd:
            self.n_episode += 1.
        self.replaybuffer.add(s, a, sp, r, gamma)
        self.n_samples += 1
        #self.agent_function.test_sparse(self.replaybuffer, self.n_samples)
        if self.replaybuffer.getSize() >= self.warm_up_steps:
            self.start_learning = True
            for _ in range(self.planningSteps):
                bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.batchSize)
                self.agent_function.train(bs, ba, bsp, br, bgamma)