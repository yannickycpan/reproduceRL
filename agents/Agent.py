import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from utils.replaybuffer import RecencyBuffer as recbuff
from utils.replaybuffer import StateBuffer as staterecbuff
import numpy as np
import math
from utils.statistics import update_stats


default_params = {'trainFrequency': 1, 'targetUpdateFrequency': 1000,
                  'auxiUpdateFrequency': 1000, 'record_every': 100000,
                  'model_n_h1': 64, 'model_n_h2': 64,
                    'n_h1': 64, 'n_h2': 64,
                  'qnntype': 'Regular',
                  'warmUpSteps': 5000, 'numGradientSteps': 100,
                  'epsilon': 1.0, 'queue_batchSize': 16,
                  'epsilonDecay': 0.0, 'epsilonMin': 0.1,
                  'gamma': 1.0, 'name': 'NoName', 'gamma_thres': -1,
                  'envName': 'NoName',
                  'kaap': 1.0, 'varianceFactor': 0.1,
                  'batchSize': 32, 'auxibatchSize': 64, 'bufferSize': 1000000,
                  'queueSize': 1000000, 'planningSteps': 1, 'numAuxUpdate': 1,
                  'saveModel': True, 'useSavedModel': False,
                  'useTrueModel': True, 'notTrain': False,
                  'type': 'regular',
                  'useAtari': False, 'tau': 0.001, 'critic_factor': 10.0,
                    'stateBounded': False,  'stateBound': None,
                  'noiseScale': 0.1, 'sparseReward': False,
                  'modelInfo': None, 'actionBound': None,
                  'stateLow': None, 'stateHigh': None,
                  'percentileThres': 75, 'scaleState4Linear': True,
                  'lam': 0.95, 'resetFrequency': 10000,
                  'forgetRate': -1, 'radius': 10.0,
                  'useStretch': 0, 'liftProjectState': 0,
                  'maxgaloops': 200, 'numSCsamples': 20, 'search_control_frequency': 1,
                  'startTravel': 0, 'stopTravel': 1000000,
                  'gaAlpha': 0.01, 'meta_learning_rate': 0.001, 'alpha': 0.0001, 'policy_alpha': 0.00001,
                    'model_learning_rate': 0.0001,
                  'innerAlpha': 0.1, 'critic_times': 1,
                  'planDepth': 5, 'useRollout': 0, 'useTargetHC': 1,
                  'mixwithValue': 0, 'priorityScale': 1.0,
                  'useGAE': 1, 'ppoClipRatio': 0.1, 'klDelta': 0.01, 'entropy_reg': 0.0,
                  'PSDreg': 1.0, 'Anormreg': 1.0, 'decor_reg': 0.0,
                  'model_loss_weight': 1.0, 'phi_norm': 0.0, 'model_reg_weight': 0.0, 'F_learning_rate': 0.0,
                  'trainR': False, 'cotrain_phisp': False, 'action_model': False,
                  'reward_loss_weight': 0.0, 'useTarNN_phisp': True,
                    # the two are identical, modify later
                  'reward_scale': 1.0, 'statescale': 1.0,
                  # hidden layer type, assume it is either relu or tanh units
                  'usetanh': 0, 'allsparseact': False,
                  # sparse_dim will be calculated automatically unless test_tiling mode is true
                 'n_tiles': 20, 'extra_strength': False, 'n_tilings': 1, 'sparse_dim': None,
                 'test_tiling': False, 'actfunctypeFTA': 'linear', 'actfunctypeFTAstrength': 'linear',
                  'fta_input_max': 1.0, 'outofbound_reg': 0.0,
                  'self_strength': False, 'sparseactor': 0, 'spexpscalor': 0.05,
                  'spexpscaloradd': 0.05, 'continoisescale': 1.0,
                  # this has effect only when use cartpole, temp is used in linear actor critic softmax
                  'use_extreme_init': True, 'temperature': 1.0}


''' sparseacttype: act_func_sparse = [tf.nn.tanh, None, tf.nn.relu] '''

def merge(params):
    for key in default_params:
        if key not in params:
            params[key] = default_params[key]
    return params

class Agent(object):
    def __init__(self, params):
        params = merge(params)
        self.name = params['name']
        ''' mini-batch training related '''
        self.maxTotalSamples = params['maxTotalSamples']
        self.replaybuffer = recbuff(params['bufferSize'])
        self.modelreplaybuffer = recbuff(params['bufferSize'])
        self.prioritizedreplaybuffer = staterecbuff(params['bufferSize'])
        self.priority_scale = params['priorityScale']

        self.bufferSize = params['bufferSize']
        self.batchSize = params['batchSize']
        self.planningSteps = params['planningSteps']
        self.queue_batchSize = params['queue_batchSize']
        self.er_batchSize = int(self.batchSize - self.queue_batchSize)
        self.warm_up_steps = params['warmUpSteps']

        self.trainFrequency = params['trainFrequency']
        self.scaleState4Linear = params['scaleState4Linear']
        self.forgetRate = params['forgetRate']

        ''' environment info '''
        self.env_name = params['envName']
        self.stateDim = params['stateDim']
        self.actionDim = params['actionDim']
        self.actionBound = params['actionBound']
        self.stateBounded = params['stateBounded']
        self.use_atari_nn = params['useAtari']

        ''' exploration noise '''
        self.epsilon = params['epsilon']
        self.epsilonDecay = params['epsilonDecay']
        self.epsilonMin = params['epsilonMin']
        self.noise_t = np.zeros(self.actionDim)
        self.conti_noisescale = params['continoisescale']

        ''' discount rate and boostrap parameter '''
        self.gamma = params['gamma']
        self.lam = params['lam']

        ''' used in model-based methods , TODO: change the key to the same as variable name '''
        self.stop_traveling = params['stopTravel']
        self.start_traveling = params['startTravel']
        self.search_control_frequency = params['search_control_frequency']
        self.noise_scale = params['noiseScale']
        self.num_sc_samples = params['numSCsamples']
        self.planDepth = params['planDepth']
        self.useRollout = params['useRollout']
        self.saveModel = params['saveModel']
        self.useSavedModel = params['useSavedModel']
        self.useTrueModel = params['useTrueModel']
        self.modelinfo = params['modelInfo']
        self.num_aux_udpate_per_time = params['numAuxUpdate']
        self.termination_conditions = self.modelinfo.termination_condition
        self.true_env_model = self.modelinfo.model if self.modelinfo is not None else None
        self.model_query = self.env_model_query if self.useTrueModel else self.learned_model_query
        print(' use true model is ==================================== ', self.useTrueModel)

        self.sparseReward = params['sparseReward']
        self.atleast_one_succ = False

        ''' other usage '''
        self.notTrain = params['notTrain']
        self.atleast_one_succ = False
        self.n_episode = 0.0
        self.n_samples = 0
        self.start_learning = False

        self.agent_function = None
        self.mode = None

        ''' useful statistics information '''
        self.useful_statistics(params)

        ''' special utils '''
        if self.env_name == 'GridWorld':
            from utils.mathutils import cartesian_product_simple_transpose
            self.custermized_log = self.gd_custermized_log
            self.samples_for_est = 3000
            self.gd_all_states = cartesian_product_simple_transpose([np.linspace(0., 1., 50), np.linspace(0., 1., 50)])

    def useful_statistics(self, params):
        if isinstance(self.stateDim, list):
            print(' image input is used !!!!!!!!!!!!!!!!!!!!!!!!!!!!! line 132 in Agent.py')
            return
        self.s_lowb = params['stateLow']
        self.s_upb = params['stateHigh']
        '''needed model parameters'''
        self.mu_s = np.zeros(self.stateDim)
        self.covmat_s = np.eye(self.stateDim)
        self.covmat_s_first = np.zeros((self.stateDim, self.stateDim))

        self.mu_diff = np.zeros(self.stateDim)
        self.covmat_diff = np.zeros((self.stateDim, self.stateDim))
        self.covmat_diff_first = np.zeros((self.stateDim, self.stateDim))
        self.covmat_diff_inv = np.zeros((self.stateDim, self.stateDim))
        self.diff_weight = np.ones((1, self.stateDim))

        self.mean_s_diff = 0.0
        self.empirical_s_upb = None
        self.empirical_s_lowb = None
        ''' we learn a bound only for those unbounded state spaces '''
        if params['envName'] in ['CartPole-v1']:
            self.s_upb[1], self.s_upb[-1] = 10., 10.
            self.empirical_s_upb = self.s_upb
            self.empirical_s_lowb = -self.empirical_s_upb
        elif params['envName'] in ["Walker2d-v2", "Hopper-v2", "Swimmer-v2",
                                   "InvertedPendulum-v2", "Reacher-v2", "HalfCheetah-v2"]:
            self.s_upb = self.empirical_s_lowb = -5. * np.ones(self.stateDim)
            self.s_lowb = self.empirical_s_upb = 5. * np.ones(self.stateDim)
        else:
            self.empirical_s_upb = self.s_upb
            self.empirical_s_lowb = self.s_lowb
        self.move_thres = None

    def linear_decay_epsilon(self):
        if self.start_learning and self.use_atari_nn:
            self.epsilon = max(self.epsilonMin, self.epsilon - self.epsilonDecay)

    def take_action(self, state):
        if np.random.uniform(0.0, 1.0) < self.epsilon or not self.start_learning:
            action = np.random.randint(self.actionDim)
        else:
            action = self.agent_function.take_action(state)
        return action

    def take_action_eval(self, state, epsilon=0.05):
        action = self.agent_function.take_action(state)
        if self.actionBound is not None:
            return np.clip(action, -self.actionBound, self.actionBound)
        else:
            if np.random.uniform(0.0, 1.0) < epsilon:
                return np.random.randint(self.actionDim)
            action = self.agent_function.take_action(state)
            return action

    def _scale(self, state):
        if self.env_name == 'MountainCar-v0':
            state = (state - np.array([-1.2, -0.07]))/(np.array([0.6, 0.07]) - np.array([-1.2, -0.07]))
            return state
        return state

    ''' should be called BEFORE increasing sample count self.n_samples '''
    def update_statistics(self, s, sp):
        if 'GridWorld' not in self.env_name:
            self.mu_s, self.covmat_s_first, self.covmat_s \
                = update_stats(s, self.mu_s, self.covmat_s_first, self.n_samples)
        self.mean_s_diff = (self.mean_s_diff * self.n_samples + np.linalg.norm(sp - s, ord=2)) / (self.n_samples + 1.)
        self.move_thres = self.mean_s_diff / np.sqrt(self.stateDim)
        #if self.env_name in ['CartPole-v1']:
        #    self.empirical_s_upb = np.maximum(self.empirical_s_upb, np.abs(s))
        #    self.empirical_s_lowb = -self.empirical_s_upb

    def update_stats_diff_s(self, diff):
        self.mu_diff, self.covmat_diff_first, self.covmat_diff \
            = update_stats(diff, self.mu_diff, self.covmat_diff_first, self.n_samples)
        if self.n_samples % 1000 == 0 and self.n_samples > 0:
            self.diff_weight[0, :] = 1.0/(self.covmat_diff.diagonal() + np.ones(self.stateDim)*0.00001)
            self.diff_weight[0, :] = self.diff_weight[0, :]/np.linalg.norm(self.diff_weight, ord=2)

    def update(self, s, a, sp, r, episodeEnd, info):
        if self.notTrain:
            return
        gamma = self.gamma if not episodeEnd else 0.0
        self.linear_decay_epsilon()
        self.replaybuffer.add(s, a, sp, r, gamma)
        self.n_samples += 1
        if self.replaybuffer.getSize() >= self.warm_up_steps and self.n_samples % self.trainFrequency == 0:
            for pn in range(self.planningSteps):
                bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.batchSize)
                self.agent_function.train(bs, ba, bsp, br, bgamma)
                self.start_learning = True


    def stack_two_batches(self, model_samples, er_samples):
        stacked = []
        for hats, ers in zip(model_samples, er_samples):
            stacked.append(np.squeeze(np.vstack([hats, ers])))
        return stacked

    def env_model_query(self, ss, aa):
        if len(ss.shape) == 1:
            sp, r, gamma = self.true_env_model(ss, aa)
            gamma = self.gamma if gamma is None else gamma
            return sp[None, :], np.array([r]), np.array([gamma])
        spp, rr, gg = [], [], []
        for (s, a) in zip(ss, aa):
            sp, r, gamma = self.true_env_model(s, a)
            spp.append(sp)
            rr.append(r)
            gamma = self.gamma if gamma is None else gamma
            gg.append(gamma)
        return np.vstack(spp), np.array(rr), np.array(gg)

    def get_p_star(self, gd_states):
        ba = self.agent_function.take_action(gd_states)
        bsp, br, bg = self.env_model_query(gd_states, ba)
        priorities = self.agent_function.td_scale(gd_states, ba, bsp, br, bg)
        priorities += 1e-7
        prob = priorities/np.sum(priorities)
        return prob.reshape((50, 50))

    def gd_custermized_log(self, logger):
        ''' temporarily not in use '''
        # return
        if 'UniformDist' not in logger.logger_dict:
            logger.logger_dict['UniformDist'] = []
        if 'OnPolicyDist' not in logger.logger_dict:
            logger.logger_dict['OnPolicyDist'] = []
        if 'OnPolicyBaseDist' not in logger.logger_dict:
            logger.logger_dict['OnPolicyBaseDist'] = []
        if 'UniformBaseDist' not in logger.logger_dict:
            logger.logger_dict['UniformBaseDist'] = []
        p_hat = self.get_p_hat()
        if p_hat is None:
            placeholder = 1.0
            logger.logger_dict['UniformDist'].append(placeholder)
            logger.logger_dict['OnPolicyDist'].append(placeholder)
            logger.logger_dict['OnPolicyBaseDist'].append(placeholder)
            logger.logger_dict['UniformBaseDist'].append(placeholder)
            return
        p_star = self.get_p_star(self.gd_all_states)
        weight = self.get_on_policy_weight()
        uniformdist = np.mean(np.abs(p_star - p_hat))
        onpolicydist = np.sum(weight * np.abs(p_star - p_hat))
        baseonpolicydist = np.sum(weight * np.abs(p_star - 1./2500))
        baseonuniformdist = np.mean(np.abs(p_star - 1./2500))

        logger.logger_dict['UniformDist'].append(uniformdist)
        logger.logger_dict['OnPolicyDist'].append(onpolicydist)
        logger.logger_dict['OnPolicyBaseDist'].append(baseonpolicydist)
        logger.logger_dict['UniformBaseDist'].append(baseonuniformdist)
        print(' the uniformDist and OnPolicyDist are :: ----------------- : ', uniformdist, onpolicydist)

    def get_visit_index(self, states, resolution=0.02):
        visit_count = np.zeros((50, 50))
        for i in range(states.shape[0]):
            xind, yind = min(49, math.floor(states[i, 0] / resolution)), min(49, math.floor(states[i, 1] / resolution))
            visit_count[xind, yind] += 1
        return visit_count

    def get_on_policy_weight(self):
        bs, ba, bsp, br, bg = self.replaybuffer.sample_batch(self.samples_for_est)
        visit_count = self.get_visit_index(bs)
        return visit_count/np.sum(visit_count)

    def get_p_hat(self):
        return

    def learned_model_query(self, ss, aa):
        raise NotImplementedError

    def custermized_log(self, logger):
        return

    def save_model(self, env_name, agent_name):
        if self.saveModel:
            self.agent_function.save_model(env_name, agent_name)
        return None
