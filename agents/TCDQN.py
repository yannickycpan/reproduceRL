import tensorflow as tf
import numpy as np
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from FunctionApproximator import FunctionApproximator
from network.dqn_network import create_fta_qnn
from network.operations import compute_q_acted, update_target_nn_assign, update_critic_ops, listshape2vec
from Agent import Agent
from network.ftann import FTA
from sparseutils import tfget_sparsities,  get_grad_interferences


class DQN(FunctionApproximator):
    def __init__(self, params):
        super(DQN, self).__init__(params)

        self.create_qnn = create_fta_qnn

        self.unittype = tf.nn.tanh if self.usetanh == 1 else tf.nn.relu
        self.statescale = 255. if self.use_atari_nn else 1.0
        print('check if use atari nn or not -------------------- ', self.use_atari_nn)

        if 'NoTarget' in self.agent_name:
            self.update_target_frequency = 1
            self.useTargetHC = 0
            self.compute_Qtarget = self.compute_Qtarget_wotar
        else:
            self.compute_Qtarget = self.compute_Qtarget_wtar

        print('statedim, actiondim, tiles, use atari, target nn move rate are ---------------------------------- ',
              self.stateDim, self.actionDim, self.create_qnn, self.update_target_frequency,
              self.use_atari_nn, self.statescale)
        # params['sess'] = self.sess
        with self.g.as_default():
            self.fta = FTA(params)
            '''used for batch normalization'''
            self.action_input = tf.placeholder('int64', [None])
            self.state_input, self.q_values, self.max_q_value, self.best_act, self.phi, self.tvars = \
                self.create_qnn(self.agent_name, self.dtype, self.stateDim, self.actionDim, self.n_h1, self.n_h2,
                                self.unittype, self.fta)

            self.tar_state_input, self.tar_q_values, self.tar_max_q_value, self.tar_best_act, self.tar_phi, self.tar_tvars \
                = self.create_qnn("target_"+self.agent_name, self.dtype, self.stateDim, self.actionDim, self.n_h1, self.n_h2,
                                  self.unittype, self.fta)
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
            self.gradvars_list = \
                tf.gradients(tf.reduce_mean(tf.square(self.qtarget_input - tf.squeeze(self.sa_value))), self.tvars)
            self.flattened_gradvec = listshape2vec(self.gradvars_list)
            self.flattened_hidden2vec = tf.reshape(self.gradvars_list[2], [-1])
            print('flattened 2nd hidden layer training parameter shape is :: ', self.flattened_hidden2vec.shape)
            # initialize network
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.target_params_update)
            self.saver = tf.train.Saver()

            self.sparse_dim = int(self.phi.shape[1])

    def compute_Qtarget_wtar(self, sp, r, gamma):
        max_q = self.sess.run(self.tar_max_q_value, feed_dict={self.tar_state_input: sp})
        qtarget = r + gamma * max_q
        return qtarget

    def compute_Qtarget_wotar(self, sp, r, gamma):
        max_q = self.sess.run(self.max_q_value, feed_dict={self.state_input: sp})
        qtarget = r + gamma * max_q
        return qtarget

    def take_action(self, state):
        if len(state.shape) < len(self.state_input.shape):
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
        #if self.update_count % 500 == 0:
        #    print(' the mean bound is ------------------------------- ', self.get_state_dependent_bound(s))
        #    self.test_gradnorm(s, a)
        self.update_count += 1

    def get_params_grad_interference(self, replaybuffer, batchSize):
        s, a, sp, r, g = replaybuffer.sample_batch(batchSize)
        vecs_all = []
        vecs_2nd = []
        qtarget = self.compute_Qtarget(sp / self.statescale, r / self.reward_scale, g)
        for i in range(s.shape[0]):
            vec_all, vec_2nd = self.sess.run([self.flattened_gradvec, self.flattened_hidden2vec],
                                feed_dict={self.state_input: s[[i], :] / self.statescale,
                                           self.qtarget_input: qtarget[[i]],
                                           self.action_input: a[[i]]})
            vecs_all.append(vec_all)
            vecs_2nd.append(vec_2nd)
        vecs_all = np.vstack(vecs_all)
        vecs_2nd = np.vstack(vecs_2nd)
        print('the shape is ========================= ', vecs_2nd.shape, vecs_all.shape)
        gradallparams_dict = get_grad_interferences(vecs_all)
        grad2ndparams_dict = get_grad_interferences(vecs_2nd, '2ndlayer')
        gradallparams_dict.update(grad2ndparams_dict)
        return gradallparams_dict


class TCDQNAgent(Agent):
    def __init__(self, params):
        super(TCDQNAgent, self).__init__(params)

        self.agent_function = DQN(params)

    def take_action(self, state):
        if np.random.uniform(0.0, 1.0) < self.epsilon or not self.start_learning:
            action = np.random.randint(self.actionDim)
        else:
            action = self.agent_function.take_action(state)
        return action

    def custermized_log(self, logger):
        if 'OverlapSparse' not in logger.logger_dict:
            logger.logger_dict['OverlapSparse'] = []
        if 'InstanceSparse' not in logger.logger_dict:
            logger.logger_dict['InstanceSparse'] = []
        if 'GradDot' not in logger.logger_dict:
            for key in ['GradDot', 'NegGradDot', 'NegGradDotProp', 'InstanceGradSparse']:
                logger.logger_dict[key] = []
                logger.logger_dict[key + '2ndlayer'] = []
        if self.replaybuffer.getSize() >= self.batchSize:
            overlapsp, instancesp = tfget_sparsities(self.agent_function.sess, self.agent_function.state_input,
                                                     self.agent_function.phi, self.replaybuffer,
                                                     self.agent_function.batchSize, self.agent_function.statescale)
            logger.logger_dict['OverlapSparse'].append(overlapsp)
            logger.logger_dict['InstanceSparse'].append(instancesp)
            gradinfodict \
                = self.agent_function.get_params_grad_interference(self.replaybuffer, self.batchSize)
            for key in gradinfodict:
                logger.logger_dict[key].append(gradinfodict[key])
            print(' the overlap and instance are :: ==================== ',
                  overlapsp, instancesp, gradinfodict)

    def update(self, s, a, sp, r, episodeEnd, info):
        if self.notTrain:
            return
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