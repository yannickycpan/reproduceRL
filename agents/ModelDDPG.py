import tensorflow as tf
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from FunctionApproximator import FunctionApproximator
from network.actor_critic_network import create_actor_nn, create_critic_nn
from network.operations import update_target_nn_assign, update_target_nn_move, compute_normsquare_gradient_hessian
from Agent import Agent
from hillclimbingutils import *
from utils.replaybuffer import StateBuffer as staterecbuff

class DDPG(FunctionApproximator):
    def __init__(self, params):
        # NOTE the first assignment
        super(DDPG, self).__init__(params)
        # use the same setting as used in the paper
        self.actor_lc = params['alpha']
        self.noise_scale = params['noiseScale']
        self.ga_learning_rate = params['gaAlpha']
        self.critic_lc = params['critic_factor'] * params['alpha']

        self.maxgaloops = params['maxgaloops']
        self.num_sc_samples = params['numSCsamples']
        self.useTargetHC = params['useTargetHC']
        self.name = params['name']

        with self.g.as_default():
            # create actor NN and target actor NN
            self.actor_input, self.actor_output, self.actor_vars \
                = create_actor_nn('actor', self.stateDim, self.actionDim, self.actionBound, self.n_h1, self.n_h2)
            self.target_actor_input, self.target_actor_output, self.target_actor_vars \
                = create_actor_nn('target_actor', self.stateDim, self.actionDim, self.actionBound, self.n_h1, self.n_h2)
            # create critic NN and target critic NN
            self.critic_input_s, self.critic_input_a, self.critic_output, self.critic_vars \
                = create_critic_nn('critic', self.stateDim, 1, self.actor_output, self.n_h1, self.n_h2, self.usetanh)
            self.target_critic_input_s, self.target_critic_input_a, self.target_critic_output, self.target_critic_vars \
                = create_critic_nn('target_critic',  self.stateDim, 1, self.target_actor_output, self.n_h1, self.n_h2, self.usetanh)
            # create ops to update critic and actor
            self.critic_value_holders, self.critic_update, _, self.loss4ga = self.update_critic_ops(self.critic_lc)
            self.actor_update = self.update_actor_ops(self.actor_lc)
            # init target NN the same variable values
            self.tar_tvars = self.target_actor_vars + self.target_critic_vars
            self.tvars = self.actor_vars + self.critic_vars

            self.target_params_init = update_target_nn_assign(self.tar_tvars, self.tvars)
            self.target_params_update = update_target_nn_move(self.tar_tvars, self.tvars, self.tau)

            self.maxq_grad_s = tf.gradients(self.critic_output, [self.critic_input_s])[0]
            self.tar_maxq_grad_s = tf.gradients(self.target_critic_output, [self.target_critic_input_s])[0]

            _, self.grad_normsq_grad_s, self.hess_normsq_grad_s \
                = compute_normsquare_gradient_hessian('hess_grad', self.stateDim, self.maxq_grad_s,
                                                      self.critic_input_s, self.critic_output) \
                if self.useTargetHC == 0 else \
                compute_normsquare_gradient_hessian('hess_grad_tar', self.stateDim, self.tar_maxq_grad_s,
                                                         self.target_critic_input_s, self.critic_output)

            self.td_grad_s = tf.gradients(self.loss4ga, [self.critic_input_s])[0] \
                             + tf.gradients(self.loss4ga, [self.actor_input])[0]
            self.logtd_grad_s = tf.gradients(tf.math.log(self.loss4ga + 1e-5), [self.critic_input_s])[0] \
                                + tf.gradients(tf.math.log(self.loss4ga + 1e-5), [self.actor_input])[0]

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
            critic_loss_4ga = tf.reduce_mean(tf.abs(critic_value_holders - tf.squeeze(self.critic_output)))
            critic_update_grads = tf.gradients(critic_loss, self.critic_vars)
            critic_grads_feed = zip(critic_update_grads, self.critic_vars)
            critic_update = tf.train.AdamOptimizer(lr, name='critic_opt').apply_gradients(critic_grads_feed)
        return critic_value_holders, critic_update, critic_loss, critic_loss_4ga

    '''return an action to take for each state, NOTE this action is in [0, 1]'''
    def take_action(self, state):
        if len(state.shape) < 2:
            state = np.expand_dims(state, 0)
        action = self.sess.run(self.actor_output,
                               {self.actor_input: state})
        return action[0]

    def take_batch_action(self, state):
        action = self.sess.run(self.actor_output, {self.actor_input: state})
        return action

    def computeQtargets(self, state_tp, reward, gamma):
        qvalues = self.sess.run(self.target_critic_output,
                                {self.target_critic_input_s: state_tp, self.target_actor_input: state_tp})
        qtargets = reward + gamma * np.squeeze(qvalues)
        return qtargets

    ''' init_s must be an array '''
    def hc_update(self, s, model, hctype = None, prob = None):
        deltas = None
        if len(s.shape) < 2:
            s = s[None, :]
        if 'Frequency' in hctype:
            deltas = self.sess.run(self.grad_normsq_grad_s, feed_dict={self.actor_input: s,
                                                                        self.critic_input_s: s,
                                                                       self.target_actor_input: s,
                                                                       self.target_critic_input_s: s})
        elif 'HessNorm' in hctype:
            deltas = self.sess.run(self.hess_normsq_grad_s, feed_dict={self.actor_input: s,
                                                                        self.critic_input_s: s,
                                                                       self.target_actor_input: s,
                                                                       self.target_critic_input_s: s})
        elif TDHC in hctype:
            act = self.take_action(s)
            sp, r, gamma = model(np.squeeze(s), act)
            """ TD should always use target network to update parameters """
            target = self.computeQtargets(sp, r, gamma)
            nodetouse = self.td_grad_s if 'Log' not in self.name else self.logtd_grad_s
            # print(target.shape, target)
            target = np.array([target]) if len(target.shape)==0 else target
            deltas = self.sess.run(nodetouse, feed_dict={self.critic_input_s: s, self.actor_input: s,
                                                         self.critic_value_holders: target})
        elif VHC in hctype:
            deltas = self.sess.run(self.maxq_grad_s, feed_dict={self.actor_input: s,
                                                       self.critic_input_s: s})
        elif MIXTWO in hctype:
            deltalist = self.sess.run([self.grad_normsq_grad_s, self.hess_normsq_grad_s],
                                      feed_dict={self.actor_input: s,
                                                 self.critic_input_s: s,
                                                 self.target_actor_input: s,
                                                 self.target_critic_input_s: s})
            finaldelta = np.zeros(self.stateDim)
            for delta in deltalist:
                normdelta = np.linalg.norm(delta, ord=2)
                if np.isnan(normdelta) or np.isinf(normdelta) or normdelta < 0.00001:
                    continue
                #print(' the delta norm is :: ', normdelta)
                finaldelta += np.squeeze(delta)
            deltas = finaldelta
        return np.squeeze(deltas)

    def train(self, state, action, state_tp, reward, gamma):
        qtargets = self.computeQtargets(state_tp, reward, gamma)
        with self.g.as_default():
            self.sess.run(self.critic_update, feed_dict={self.critic_input_s: state, self.critic_input_a: action,
                                                         self.critic_value_holders: qtargets})
            self.sess.run(self.actor_update,
                          feed_dict={self.actor_input: state, self.critic_input_s: state})
            self.sess.run(self.target_params_update)
            self.update_count += 1
        return None


class ModelDDPGAgent(Agent):
    def __init__(self, params):
        super(ModelDDPGAgent, self).__init__(params)
        self.n_episode = 0.
        self.model_accuracy = 0.

        self.gamma_delta = 0.1

        params['n_binary'] = self.n_binary = self.modelinfo.n_binary

        algonamelist = ['ModelDDPG-ValueHC', 'ModelDDPG-LogTDHC', 'ModelDDPG-MixedQueueHC']
        hctypelist = [VHC, TDHC, VHC]
        self.scqueue_dict = {VHC: staterecbuff(params['bufferSize']),
                             MIXTWO: staterecbuff(params['bufferSize']),
                             PROPMIX: staterecbuff(params['bufferSize']),
                             TDHC: staterecbuff(params['bufferSize'])}
        self.name2q = dict(zip(algonamelist, hctypelist))

        self.agent_function = DDPG(params)
        if self.name == 'ModelDDPGRandomOnpolicy':
            self.update = self.update_baseline

    def take_action(self, state):
        if not self.start_learning:
            return np.random.uniform(-self.actionBound, self.actionBound, self.actionDim)
        action = self.agent_function.take_action(state)
        self.noise_t += np.random.normal(0.0, 0.2, action.shape) - self.noise_t * 0.15
        action = action + self.noise_t
        return np.clip(action, -self.actionBound, self.actionBound)

    def take_batch_action(self, state):
        acts = self.agent_function.take_batch_action(state)
        acts += self.noise_t + np.random.normal(0.0, 0.2, acts.shape) - self.noise_t * 0.15
        return np.clip(acts, -self.actionBound, self.actionBound)

    def test_model_accuracy(self, s, a, sp, r, g):
        hat_sp, hat_r, hat_g = self.model_query(s[None, :], a[None, :])
        self.model_accuracy += np.linalg.norm(np.squeeze(hat_sp) - sp, ord=2)
        # if self.n_samples % 1000 == 0:
            #print('state bound is------------ ', self.visited_upper_bound)
        print('real r and sample r:: -------------- ', r, hat_r)
        print('real sp and sample sp---------------- ', sp,  hat_sp)
        print('real g and sample g---------------- ', g,  hat_g)
        print('sampled sp error :: -------------- ', self.model_accuracy/float(self.n_samples))

    def add_model_sample(self, s, a):
        if not (self.start_learning
                and self.start_traveling <= self.n_samples <= self.stop_traveling
                and self.n_samples % self.search_control_frequency == 0):
            return
        n_added = ga_hc_search_control(self.replaybuffer, self.scqueue_dict, self.empirical_s_lowb, self.empirical_s_upb,
                                        self.move_thres, self.covmat_s, self.model_query, self.termination_conditions,
                                       self.agent_function, self.name2q, None)
        return n_added

    def learned_model_query(self, states, acts):
        hat_sp, hat_r, hat_g = self.agent_function.model_query(states, acts)
        # hat_sp = np.clip(hat_sp, self.empirical_s_lowb, self.empirical_s_upb)
        hat_g[hat_g < self.gamma - self.gamma_delta] = 0.0
        hat_g[hat_g >= self.gamma - self.gamma_delta] = self.gamma
        if self.termination_conditions is not None:
            hat_g = np.ones(hat_sp.shape[0])*self.gamma
            hat_g[self.termination_conditions(hat_sp)] = 0.
            # print(' termination condition is not None ==================== ')
        return hat_sp, np.squeeze(hat_r), np.squeeze(hat_g)

    def update(self, s, a, sp, r, episodeEnd, info):
        if self.notTrain:
            return
        gamma = self.gamma if not episodeEnd else 0.0
        ''' add samples to ER and Priori buffer '''
        self.replaybuffer.add(s, a, sp, r, gamma)
        self.update_statistics(s, sp)
        self.n_samples += 1
        self.add_model_sample(s, a)
        #if self.n_samples % 500 == 0:
        #    self.test_model_accuracy(s, a, sp, r, gamma)
        if self.replaybuffer.getSize() >= int(4*self.batchSize) \
                and self.n_samples < self.stop_traveling and not self.useTrueModel:
            bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(int(4*self.batchSize))
            self.agent_function.train_env_model(bs, ba, bsp, br, bgamma)
        if self.replaybuffer.getSize() >= self.warm_up_steps and self.n_samples % self.trainFrequency == 0:
            self.start_learning = True
            for _ in range(self.planningSteps):
                if self.start_traveling <= self.n_samples <= self.stop_traveling \
                        and self.scqueue_dict[self.name2q[self.name]].getSize() >= self.queue_batchSize:
                    hat_s = self.scqueue_dict[self.name2q[self.name]].sample_batch(self.queue_batchSize)
                    acts = self.take_batch_action(hat_s)
                    hat_sp, hat_r, hat_g = self.model_query(hat_s, acts)
                    bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.er_batchSize)
                    bs, ba, bsp, br, bgamma = self.stack_two_batches([hat_s, acts, hat_sp, hat_r[:, None], hat_g[:, None]],
                                                                     [bs, ba, bsp, br[:, None], bgamma[:, None]])
                    self.agent_function.train(bs, ba, bsp, br, bgamma)
                    continue
                bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.batchSize)
                self.agent_function.train(bs, ba, bsp, br, bgamma)

    ''' verify the on-policy sampling actions '''
    def update_baseline(self, s, a, sp, r, episodeEnd, info):
        if self.notTrain:
            return
        gamma = self.gamma if not episodeEnd else 0.0
        ''' add samples to ER and Priori buffer '''
        self.replaybuffer.add(s, a, sp, r, gamma)
        self.n_samples += 1
        #if self.n_samples % 1000 == 0:
        #    print('running baseline here !!! ')
        if self.replaybuffer.getSize() >= int(self.batchSize*4) \
                and self.n_samples < self.stop_traveling and not self.useTrueModel:
            bs, ba, bsp, br, _ = self.replaybuffer.sample_batch(int(4*self.batchSize))
            self.agent_function.train_env_model(bs, ba, bsp, br, self.diff_weight)
        if self.replaybuffer.getSize() >= self.warm_up_steps and self.n_samples % self.trainFrequency == 0:
            self.start_learning = True
            for _ in range(self.planningSteps):
                hat_s, _, _, _, _ = self.replaybuffer.sample_batch(self.queue_batchSize)
                acts = self.take_batch_action(hat_s)
                hat_sp, hat_r, hat_g = self.model_query(hat_s, acts)
                bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.er_batchSize)
                bs, ba, bsp, br, bgamma = self.stack_two_batches([hat_s, acts, hat_sp, hat_r[:,None], hat_g[:,None]],
                                                                 [bs, ba, bsp, br[:,None], bgamma[:, None]])
                self.agent_function.train(bs, ba, bsp, br, bgamma)

    def stack_two_batches(self, model_samples, er_samples):
        stacked = []
        count = 0
        for hats, ers in zip(model_samples, er_samples):
            stackedtemp = np.vstack([hats, ers]) if count < 3 else np.squeeze(np.vstack([hats, ers]))
            stacked.append(stackedtemp)
            count += 1
        return stacked