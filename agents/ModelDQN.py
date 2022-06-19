import tensorflow as tf
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from FunctionApproximator import FunctionApproximator
from network.dqn_network import create_qnn
from network.operations import compute_q_acted, update_target_nn_assign, compute_normsquare_gradient_hessian
from Agent import Agent
from hillclimbingutils import *
from utils.replaybuffer import StateBuffer as staterecbuff


class ModelDQN(FunctionApproximator):
    def __init__(self, params):
        super(ModelDQN, self).__init__(params)

        self.ga_learning_rate = params['gaAlpha']
        self.noise_scale = params['noiseScale']
        self.maxgaloops = params['maxgaloops']
        self.stateBounded = params['stateBounded']
        self.num_sc_samples = params['numSCsamples']
        self.useTargetHC = params['useTargetHC']
        self.mixwithV = params['mixwithValue']
        self.name = params['name']

        self.unittype = tf.nn.tanh if self.usetanh == 1 else tf.nn.relu
        print(' activation function used is ------------------------ ', self.unittype)

        self.allqueue_type = ['Value', 'HessNorm', 'Frequency', 'TD']
        self.ga_norm_mean = 0.
        self.ga_count = 0.
        self.create_qnn = create_qnn

        print('statedim, actiondim, use atari are ---------------------------------- ',
              self.stateDim, self.actionDim, self.create_qnn)

        with self.g.as_default():
            '''used for batch normalization'''
            self.is_training = tf.placeholder(tf.bool, [])
            self.action_input = tf.placeholder('int64', [None])
            self.state_input, self.q_values, self.max_q_value, self.best_act, self.tvars = \
                self.create_qnn("qnn", self.dtype, self.stateDim, self.actionDim, self.n_h1, self.n_h2, self.unittype)
            self.tar_state_input, self.tar_q_values, self.tar_max_q_value, self.tar_best_act, self.tar_tvars \
                = self.create_qnn("target_qnn", self.dtype, self.stateDim, self.actionDim, self.n_h1, self.n_h2, self.unittype)
            # define state action value
            self.sa_value, self.tar_sa_value \
                = compute_q_acted("sa_values", self.action_input, self.actionDim, self.q_values, self.tar_q_values)
            # define loss operation
            self.qtarget_input, self.loss, self.loss4ga, self.logloss4ga = self.define_loss("losses")
            # define optimization
            self.params_update = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # update target network
            self.target_params_update = update_target_nn_assign(self.tar_tvars, self.tvars)
            # gradient w.r.t state
            self.grad_s = tf.gradients(self.sa_value, [self.state_input])[0]
            self.td_grad_s = tf.gradients(self.loss4ga, [self.state_input])[0]
            self.logtd_grad_s = tf.gradients(self.logloss4ga, [self.state_input])[0]

            self.maxq_grad_s = tf.gradients(self.max_q_value, [self.state_input])[0]
            self.tar_maxq_grad_s = tf.gradients(self.tar_max_q_value, [self.tar_state_input])[0]

            self.mixprod_grad_s, self.grad_normsq_grad_s, self.hess_normsq_grad_s \
                = compute_normsquare_gradient_hessian('hess_grad', self.stateDim, self.maxq_grad_s,
                                                      self.state_input, self.max_q_value) \
                if self.useTargetHC == 0 else\
                compute_normsquare_gradient_hessian('hess_grad_tar', self.stateDim, self.tar_maxq_grad_s,
                                                    self.tar_state_input, self.max_q_value)

            # initialize network
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.target_params_update)

    def define_loss(self, scopename):
        with tf.variable_scope(scopename):
            q_targets = tf.placeholder(self.dtype, [None])
            loss = tf.losses.mean_squared_error(q_targets, self.sa_value)
            loss4ga = tf.losses.mean_squared_error(q_targets, self.max_q_value)
            logloss4ga = tf.math.log(tf.reduce_mean(tf.abs(q_targets - self.max_q_value)) + 1e-7)
            # logloss4ga = tf.math.log(tf.reduce_mean(tf.abs(q_targets - self.max_q_value)) + 1e-5)
            # similar to the above
        return q_targets, loss, loss4ga, logloss4ga

    def compute_Qtarget(self, sp, r, gamma):
        with self.g.as_default():
            max_q = self.sess.run(self.tar_max_q_value, feed_dict={self.tar_state_input: sp})
            qtarget = r + gamma * max_q
            return qtarget

    def compute_diff_q_target(self, sp, r, gamma):
        with self.g.as_default():
            max_q = self.sess.run(self.max_q_value, feed_dict={self.state_input: sp[:int(sp.shape[0]/2), :]})
            tar_max_q = self.sess.run(self.tar_max_q_value, feed_dict={self.tar_state_input: sp[int(sp.shape[0]/2):, :]})
            all_max_q = np.concatenate([max_q, tar_max_q])
            qtarget = r + gamma * all_max_q
            return qtarget

    def compute_current_q(self, s, r, gamma):
        max_q = self.sess.run(self.max_q_value, feed_dict={self.state_input: s})
        qtarget = r + gamma * max_q
        return qtarget

    ''' init_s must be an array '''
    def hc_update(self, s, model, hctype = None, prob = None):
        deltas = None
        if len(s.shape) < 2:
            s = s[None, :]
        #if 'Frequency' in hctype:
        #    deltas = self.sess.run(self.grad_normsq_grad_s, feed_dict={self.state_input: s, self.tar_state_input: s})
        #elif 'HessNorm' in hctype:
        #    deltas = self.sess.run(self.hess_normsq_grad_s, feed_dict={self.state_input: s, self.tar_state_input: s})
        if TDHC in hctype:
            ''' this action must be optimal (on-policy) '''
            act = self.take_action(s)
            sp, r, gamma = model(np.squeeze(s), act)
            """ TD should always use target network to update parameters """
            target = self.compute_current_q(sp, r, gamma) \
                if self.useTargetHC == 0 else self.compute_Qtarget(sp, r, gamma)
            nodetouse = self.td_grad_s if 'Log' not in self.name else self.logtd_grad_s
            deltas = self.sess.run(nodetouse, feed_dict={self.state_input: s,
                                                              self.qtarget_input: target})
        elif VHC in hctype:
            deltas = self.sess.run(self.maxq_grad_s, feed_dict={self.state_input: s})
        elif 'ValueHD' in hctype:
            deltas = self.sess.run(self.maxq_grad_s, feed_dict={self.state_input: s})
            deltas = -deltas
        elif MIXTWO in hctype:
            deltalist = self.sess.run([self.grad_normsq_grad_s, self.hess_normsq_grad_s],
                                      feed_dict={self.state_input: s, self.tar_state_input: s})
            finaldelta = np.zeros(self.stateDim)
            for delta in deltalist:
                normdelta = np.linalg.norm(delta, ord=2)
                if np.isnan(normdelta) or np.isinf(normdelta) or normdelta < 0.00001:
                    continue
                finaldelta += np.squeeze(delta)
            deltas = finaldelta
        elif MIXGRAD in hctype:
            deltas = self.sess.run(self.grad_normsq_grad_s,
                                      feed_dict={self.state_input: s, self.tar_state_input: s})
        elif MIXHESS in hctype:
            deltas = self.sess.run(self.hess_normsq_grad_s,
                                      feed_dict={self.state_input: s, self.tar_state_input: s})
        elif PROPMIX in hctype:
            deltalist = self.sess.run([self.grad_normsq_grad_s, self.hess_normsq_grad_s, self.maxq_grad_s],
                                      feed_dict={self.state_input: s, self.tar_state_input: s})
            deltas = np.zeros(self.stateDim)
            for id in range(len(deltalist)):
                normdelta = np.linalg.norm(deltalist[id], ord=2)
                if np.isnan(normdelta) or np.isinf(normdelta) or normdelta < 0.00001:
                    continue
                deltas += np.squeeze(deltalist[id]) * prob[id]
            #if prob[0]+prob[1] > 0.2:
            #    print('the proportions are :: ', prob)
        elif 'MixedProd' in hctype:
            deltas = self.sess.run(self.mixprod_grad_s, feed_dict={self.state_input: s, self.tar_state_input: s})
            #print(deltas.dot(covmat)/np.linalg.norm(deltas.dot(covmat)))
        elif 'Rollout' in hctype:
            act = self.take_action(s)
            s, _, _ = model(np.squeeze(s), act)
            s = np.squeeze(s)
            return s, 1.0, 1000
        return np.squeeze(deltas)

    def take_action(self, state):
        state = state[None, :] if len(state.shape) < 2 else state
        act = self.sess.run(self.best_act, feed_dict={self.state_input: state})
        act = act.reshape(-1)
        return act[0] if act.shape[0] == 1 else act

    def train(self, s, a, sp, r, gamma):
        qtarget = self.compute_Qtarget(sp, r, gamma)
        self.sess.run(self.params_update,
                      feed_dict={self.state_input: s, self.qtarget_input: qtarget, self.action_input: a})
        if self.update_count % self.update_target_frequency == 0:
            self.sess.run(self.target_params_update)
        self.update_count += 1

    def td_scale(self, s, a, sp, r, gamma):
        if len(s.shape) < len(self.state_input.shape):
            s = s[None, :]
            # a = np.array([a])
            sp = sp[None, :]
        s_value = self.sess.run(self.max_q_value, feed_dict={self.state_input: s})
        max_q = self.sess.run(self.tar_max_q_value, feed_dict={self.tar_state_input: sp})
        qtarget = r + gamma * np.squeeze(max_q)
        abstd = np.abs(qtarget - np.squeeze(s_value))
        return abstd

from utils.mathutils import cartesian_product_simple_transpose
class ModelDQNAgent(Agent):
    def __init__(self, params):
        super(ModelDQNAgent, self).__init__(params)

        self.gamma_delta = 0.1

        self.agent_function = ModelDQN(params)
        algonamelist = ['ModelDQN-TDHC', 'ModelDQN-LogTDHC', 'ModelDQN-NoisyER', 'ModelDQN-FrequencyHC',
                           'ModelDQN-ValueHC', 'ModelDQN-RolloutHC', 'ModelDQN-HessNormHC',
                           'ModelDQN-MixedSumHC', 'ModelDQN-MixedQueueHC', 'ModelDQN-ValueHD',
                        'ModelDQN-ProportionSumHC', 'ModelDQN-MixedSepQueueHC', 'ModelDQN-MixedGradQueueHC',
                        'ModelDQN-MixedHessQueueHC']
        hctypelist = [TDHC, TDHC, None, None, VHC, None, None, MIXTWO, VHC, VHC, PROPMIX, None, MIXGRAD, MIXHESS]
        self.scqueue_dict = {VHC: staterecbuff(params['queueSize']),
                             MIXTWO: staterecbuff(params['queueSize']),
                             MIXGRAD: staterecbuff(params['queueSize']),
                             MIXHESS: staterecbuff(params['queueSize']),
                             PROPMIX: staterecbuff(params['queueSize']),
                             TDHC: staterecbuff(params['queueSize'])}
        self.name2q = dict(zip(algonamelist, hctypelist))

        print('the queue size is: ======================================= ', params['queueSize'])

        if self.name in ['ModelDQNRandomOnpolicy', 'ModelDQN-Uniform']:
            self.update = self.update_baseline
            print('ModelDQN mode is baseline :: ', self.name)
        elif self.name == 'ModelDQN-MixedSepQueueHC':
            self.update = self.update_mixedsephc_dyna
            print('ModelDQN mode is mixed :: ', self.name)
        elif self.name in algonamelist:
            self.update = self.update_hc_dyna
            print('ModelDQN hc mode is :: ', self.name)
        else:
            print('updating function not found!!!! Error in line 202, ModelDQN.py file')
            exit(0)

        if self.env_name == 'GridWorld':
            self.custermized_log = self.gd_custermized_log

        if params['useSavedModel']:
            self.agent_function.restore(params['modelPath'])
            self.notTrain = True
            self.start_learning = True

    def _get_proportion(self):
        if 'Proportion' not in self.name:
            return None
        start = self.warm_up_steps
        proportion = (self.n_samples - start) / self.maxTotalSamples
        if self.n_samples < start:
            proportion = 0.
        prob = [min(0.5, proportion) / 2., min(0.5, proportion) / 2., 1.0 - min(0.5, proportion)]
        return prob

    def add_model_sample(self):
        if not (self.start_learning
                and self.start_traveling <= self.n_samples <= self.stop_traveling
                and self.n_samples % self.search_control_frequency == 0):
            return
        addrate = 0
        prob = self._get_proportion()
        if 'HC' in self.name or 'HD' in self.name:
            addrate = ga_hc_search_control(self.replaybuffer, self.scqueue_dict,
                                            self.empirical_s_lowb, self.empirical_s_upb, self.move_thres,
                                          self.covmat_s, self.model_query, self.termination_conditions,
                                           self.agent_function, self.name2q, prob=prob)
        elif self.name == 'ModelDQN-NoisyER':
            init_s, _, _, _, _ = self.replaybuffer.sample_batch(self.agent_function.num_sc_samples)
            noise = np.random.normal(np.zeros_like(init_s), np.sqrt(np.diagonal(self.covmat_s)), init_s.shape) * self.noise_scale
            ss = np.clip(init_s + noise, self.empirical_s_lowb, self.empirical_s_upb)
            if self.env_name == 'Acrobot-v1':
                ss[:2] = ss[:2] / np.linalg.norm(ss[:2], ord=2)
                ss[2:4] = ss[2:4] / np.linalg.norm(ss[2:4], ord=2)
            for i in range(ss.shape[0]):
                self.prioritizedreplaybuffer.add(ss[i, :])
        elif self.name == 'ModelDQN-Uniform':
            ss = np.random.uniform(self.empirical_s_lowb, self.empirical_s_upb, [self.num_sc_samples, self.stateDim])
            for i in range(ss.shape[0]):
                self.prioritizedreplaybuffer.add(ss[i, :])
        return addrate

    def take_batch_action(self, state):
        acts = self.agent_function.take_action(state)
        randind = (np.random.uniform(0.0, 1.0, state.shape[0]) < self.epsilon)
        acts[randind] = np.random.randint(self.actionDim, size=int(np.sum(randind)))
        return acts

    def test_model_accuracy(self, s, a, sp, r, g):
        hat_s, hat_r, hat_g = self.learned_model_query(s, a)
        #hat_s, hat_r, g = self.env_model_query(s, a)
        print('real transition is :: -------------- ', r, g)
        print('sampled transition is :: -------------- ', hat_r, hat_g)

    def learned_model_query(self, states, acts):
        hat_sp, hat_r, hat_g = self.agent_function.model_query(states, np.squeeze(acts))
        hat_sp = np.clip(hat_sp, self.empirical_s_lowb, self.empirical_s_upb)
        dones = self.termination_conditions(hat_sp) if self.termination_conditions is not None else None
        # hat_g = np.ones(hat_sp.shape[0])*self.gamma
        if len(states.shape) < 2:
            hat_g = self.gamma * (dones == False) if dones is not None else hat_g
            hat_g = 0 if hat_g < self.gamma - self.gamma_delta else self.gamma
            return hat_sp, hat_r, hat_g
        if dones is not None:
            hat_g[dones] = 0.
            return hat_sp, hat_r, hat_g
        else:
            hat_g[hat_g < self.gamma - self.gamma_delta] = 0.0
            hat_g[hat_g >= self.gamma - self.gamma_delta] = self.gamma
            return hat_sp, hat_r, hat_g

    def log_buffer_info(self):
        nsamples = 2000
        if self.n_samples % 2000 == 0 and self.scqueue_dict[self.name2q[self.name]].getSize() >= nsamples:
            plotsamples = self.scqueue_dict[self.name2q[self.name]].sample_batch(nsamples)
            np.savetxt(self.name + '_' + self.env_name + 'scqueue'+str(int(self.n_samples))+'.txt',
                       plotsamples, fmt='%10.7f', delimiter=',')

            plotsamples, _, _, _, _ = self.replaybuffer.sample_batch(nsamples)
            np.savetxt(self.name + '_' + self.env_name + 'erbuffer' + str(int(self.n_samples)) + '.txt',
                       plotsamples, fmt='%10.7f', delimiter=',')

    def count_within_circle(self, points):
        count = 0
        centers = [np.array([0.2, 0.4]), np.array([0.4, 0.9]), np.array([0.7, 0.1])]
        for center in centers:
            c = center + 0.05
            for i in range(points.shape[0]):
                if np.linalg.norm(points[i, :] - c, ord=2) < 0.1:
                    count += 1
        return count

    def custermized_log(self, logger):
        #if self.env_name == 'GridWorld':
        #    self.gd_custermized_log(logger)
        if self.env_name != 'MazeGridWorld':
            return
        if 'SCPoints' not in logger.logger_dict:
            logger.logger_dict['SCPoints'] = []
        if 'ERPoints' not in logger.logger_dict:
            logger.logger_dict['ERPoints'] = []
        samplepoints = 1000
        if self.scqueue_dict[self.name2q[self.name]].getSize() >= samplepoints:
            scsamples = self.scqueue_dict[self.name2q[self.name]].sample_batch(samplepoints)
            scpoints = self.count_within_circle(scsamples)
            ersamples, _, _, _, _ = self.replaybuffer.sample_batch(samplepoints)
            erpoints = self.count_within_circle(ersamples)
            logger.logger_dict['SCPoints'].append(scpoints)
            logger.logger_dict['ERPoints'].append(erpoints)
            print(' the sc points and er points are :: ==================== ', scpoints, erpoints)
        else:
            logger.logger_dict['SCPoints'].append(0)
            logger.logger_dict['ERPoints'].append(0)

    def get_p_star_old(self):
        bs, ba, bsp, br, bg = self.replaybuffer.sample_batch(self.samples_for_est)
        priorities = self.agent_function.td_scale(bs, ba, bsp, br, bg)
        priorities += 1e-7
        prob = priorities/np.sum(priorities)
        indexes = np.random.choice(self.samples_for_est, self.samples_for_est, replace=True, p=prob)
        bs = bs[indexes, :]
        visit_count = self.get_visit_index(bs)
        return visit_count/np.sum(visit_count)

    def get_p_hat(self):
        if self.scqueue_dict[self.name2q[self.name]].getSize() < self.samples_for_est \
                or self.replaybuffer.getSize() < self.samples_for_est:
            return None
        hat_s = self.scqueue_dict[self.name2q[self.name]].sample_batch(self.samples_for_est)
        visit_count = self.get_visit_index(hat_s)
        return visit_count/np.sum(visit_count)

    def update_hc_dyna(self, s, a, sp, r, episodeEnd, info):
        if self.notTrain:
            return
        gamma = self.gamma if not episodeEnd else 0.0
        self.update_statistics(s, sp)
        ''' add samples to ER and Priori buffer '''
        self.replaybuffer.add(s, a, sp, r, gamma)
        self.log_buffer_info()
        #if self.n_samples % 100 == 0:
        #    self.test_model_accuracy(s, a, sp, r, gamma)
        #    print(' the sizes are :: ', [self.scqueue_dict[key].getSize() for key in self.scqueue_dict])
        self.n_samples += 1.
        self.add_model_sample()
        if not self.atleast_one_succ:
            self.atleast_one_succ = episodeEnd
        ''' below is the model learning and agent learning part '''
        if self.replaybuffer.getSize() >= int(self.batchSize*4) and not self.useTrueModel:
            bs, ba, bsp, br, bg = self.replaybuffer.sample_batch(int(4*self.batchSize))
            self.agent_function.train_env_model(bs, ba, bsp, br, bg, np.ones_like(self.diff_weight))
        if self.replaybuffer.getSize() >= self.warm_up_steps and self.n_samples % self.trainFrequency == 0:
            if self.sparseReward and not self.atleast_one_succ:
                return
            self.start_learning = True
            for pn in range(self.planningSteps):
                if self.queue_batchSize > 0 and \
                        self.scqueue_dict[self.name2q[self.name]].getSize() >= self.queue_batchSize:
                    hat_s = self.scqueue_dict[self.name2q[self.name]].sample_batch(self.queue_batchSize)
                    acts = self.take_batch_action(hat_s)
                    hat_sp, hat_r, hat_g = self.model_query(hat_s, np.squeeze(acts))
                    if self.er_batchSize > 0:
                        bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.er_batchSize)
                        bs, ba, bsp, br, bgamma \
                            = self.stack_two_batches([hat_s, acts[:, None], hat_sp, hat_r[:, None], hat_g[:, None]],
                                                                     [bs, ba[:, None], bsp, br[:, None], bgamma[:, None]])
                    else:
                        bs, ba, bsp, br, bgamma = hat_s, acts, hat_sp, hat_r, hat_g
                    self.agent_function.train(bs, ba, bsp, br, bgamma)
                else:
                    bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.batchSize)
                    self.agent_function.train(bs, ba, bsp, br, bgamma)


    def update_mixedsephc_dyna(self, s, a, sp, r, episodeEnd, info):
        if self.notTrain:
            return
        gamma = self.gamma if not episodeEnd else 0.0
        self.update_statistics(s, sp)
        ''' add samples to ER and Priori buffer '''
        self.replaybuffer.add(s, a, sp, r, gamma)
        self.n_samples += 1.
        #if self.n_samples % 500 == 0:
        #    print(' the sizes are :: ', [self.scqueue_dict[key].getSize() for key in self.scqueue_dict])
        if self.start_learning and self.start_traveling <= self.n_samples <= self.stop_traveling:
            self.add_model_sample()
        if not self.atleast_one_succ:
            self.atleast_one_succ = episodeEnd
        ''' below is the model learning and agent learning part '''
        if self.replaybuffer.getSize() >= int(self.batchSize*8) and not self.useTrueModel:
            bs, ba, bsp, br, _ = self.replaybuffer.sample_batch(int(8*self.batchSize))
            self.agent_function.train_env_model(bs, ba, bsp, br, np.ones_like(self.diff_weight))
        if self.replaybuffer.getSize() >= self.warm_up_steps and self.n_samples % self.trainFrequency == 0:
            if self.sparseReward and not self.atleast_one_succ:
                return
            self.start_learning = True
            for pn in range(self.planningSteps):
                queuefull = True
                for key in [VHC, MIXTWO]:
                    queuefull = queuefull * (self.scqueue_dict[key].getSize() > self.queue_batchSize)
                if queuefull:
                    valuesc_size = self.scqueue_dict[VHC].getSize()/(self.scqueue_dict[MIXTWO].getSize()
                                                                                  + self.scqueue_dict[VHC].getSize())
                    sc_size = int(valuesc_size * self.queue_batchSize)
                    bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.er_batchSize)
                    for queuekey in [VHC, MIXTWO]:
                        if sc_size <= 0:
                            continue
                        hat_s = self.scqueue_dict[queuekey].sample_batch(sc_size)
                        acts = self.take_batch_action(hat_s)
                        hat_sp, hat_r, hat_g = self.model_query(hat_s, np.squeeze(acts))
                        bs, ba, bsp, br, bgamma = self.stack_two_batches(
                            [hat_s, acts[:, None], hat_sp, hat_r[:, None], hat_g[:, None]],
                            [bs, ba[:, None], bsp, br[:, None], bgamma[:, None]])
                        sc_size = self.queue_batchSize - sc_size
                    self.agent_function.train(bs, ba, bsp, br, bgamma)
                else:
                    bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.batchSize)
                    self.agent_function.train(bs, ba, bsp, br, bgamma)

    def projection(self, ss):
        s = ss.copy()
        if self.env_name == 'Acrobot-v1':
            s[:, :2] = s[:, :2] / np.linalg.norm(s[:, :2], ord=2, axis=1, keepdims=True)
            s[:, 2:4] = s[:, 2:4] / np.linalg.norm(s[:, 2:4], ord=2, axis=1, keepdims=True)
        return s

    def update_baseline(self, s, a, sp, r, episodeEnd, info):
        if self.notTrain:
            return
        gamma = self.gamma if not episodeEnd else 0.0
        ''' add samples to ER and Priori buffer '''
        self.replaybuffer.add(s, a, sp, r, gamma)
        self.n_samples += 1.
        if self.start_learning and self.start_traveling <= self.n_samples <= self.stop_traveling:
            self.add_model_sample()
        if not self.atleast_one_succ:
            self.atleast_one_succ = episodeEnd
        if self.replaybuffer.getSize() >= int(self.batchSize*4) and not self.useTrueModel:
            bs, ba, bsp, br, _ = self.replaybuffer.sample_batch(int(4*self.batchSize))
            self.agent_function.train_env_model(bs, ba, bsp, br, np.ones_like(self.diff_weight))
        if self.replaybuffer.getSize() >= self.warm_up_steps and self.n_samples % self.trainFrequency == 0:
            if self.sparseReward and not self.atleast_one_succ:
                return
            self.start_learning = True
            for pn in range(self.planningSteps):
                if 'Uniform' in self.name and self.prioritizedreplaybuffer.getSize() >= self.queue_batchSize:
                    hat_s = self.prioritizedreplaybuffer.sample_batch(self.queue_batchSize)
                else:
                    hat_s, _, _, _, _ = self.replaybuffer.sample_batch(self.queue_batchSize)
                acts = self.take_batch_action(hat_s)
                hat_sp, hat_r, hat_g = self.model_query(hat_s, np.squeeze(acts))
                if self.queue_batchSize < self.batchSize:
                    bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.er_batchSize)
                    bs, ba, bsp, br, bgamma = self.stack_two_batches(
                        [hat_s, acts[:, None], hat_sp, hat_r[:, None], hat_g[:, None]],
                        [bs, ba[:, None], bsp, br[:, None], bgamma[:, None]])
                else:
                    bs, ba, bsp, br, bgamma = hat_s, acts, hat_sp, hat_r, hat_g
                self.agent_function.train(bs, ba, bsp, br, bgamma)
