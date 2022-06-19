import tensorflow as tf
import datetime
import os
import sys
import numpy as np
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from network.model_network import create_discrete_action_model, create_nobinary_continuous_action_model
from network.operations import model_loss, extract_3d_tensor, extracted_2d_tensor
from utils.statistics import update_stats


class FunctionApproximator(object):
    def __init__(self, params):
        self.seed = params['seed']

        self.learning_rate = params['alpha']
        self.model_learning_rate = params['model_learning_rate']
        self.meta_learning_rate = params['meta_learning_rate']
        self.env_name = params['envName']
        self.agent_name = params['name']
        self.useTrueModel = params['useTrueModel']
        self.learn_env_model = not self.useTrueModel
        self.reward_scale = params['reward_scale']

        self.batchSize = params['batchSize']

        ''' for HC MBRL '''
        self.noise_scale = params['noiseScale']
        self.ga_learning_rate = params['gaAlpha']
        self.maxgaloops = params['maxgaloops']
        self.num_sc_samples = params['numSCsamples']
        self.useTargetHC = params['useTargetHC']
        self.avg_addrate = 0.0

        ''' for batch Policy Gradient method '''
        self.use_gae = params['useGAE']
        self.kl_delta = params['klDelta']
        self.ppo_clip_ratio = params['ppoClipRatio']

        ''' other common settings '''
        self.n_h1 = params['n_h1']
        self.n_h2 = params['n_h2']

        self.usetanh = params['usetanh']

        # target network moving rate
        self.tau = params['tau']

        self.dtype = tf.float32
        self.stateDim = params['stateDim']
        self.actionDim = params['actionDim']
        self.actionBound = params['actionBound']

        self.use_atari_nn = params['useAtari']

        self.update_target_frequency = params['targetUpdateFrequency']
        self.auxiUpdateFrequency = params['auxiUpdateFrequency']
        self.num_gradient_steps = params['numGradientSteps']
        self.update_count = 0
        self.n_binary = 0

        self.count = 0

        self.shapelist = []
        self.sizelist = []
        self.n_total_vars = 0

        print('target update frequency and train frequency are and usetanh are ============ ',
              self.update_target_frequency, params['trainFrequency'], self.usetanh)

        self.g = tf.Graph()

        # this variable needs to be initialized in subclass
        self.saver = None

        with self.g.as_default():
            tf.set_random_seed(self.seed)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            '''model based RL setting'''
            if self.learn_env_model and self.actionBound is None:
                self.model_n_h1 = params['model_n_h1']
                self.model_n_h2 = params['model_n_h2']
                self.model_s_input, self.model_sp_diff, self.model_r, self.model_gamma\
                    = create_discrete_action_model('envmodel', self.stateDim, self.actionDim,
                                                      self.model_n_h1, self.model_n_h2)
                self.model_a_input = tf.placeholder(tf.int64, [None])

                self.query_sp_diff = extract_3d_tensor(self.model_sp_diff, self.model_a_input)
                self.query_sp = self.query_sp_diff + self.model_s_input
                self.query_r = extracted_2d_tensor(self.model_r, self.model_a_input)
                self.query_gamma = extracted_2d_tensor(self.model_gamma, self.model_a_input)

                self.model_sp_diff_target, self.model_r_target, self.model_gamma_target, self.model_loss\
                    = model_loss('model_loss', self.stateDim, self.query_sp_diff, self.query_r, self.query_gamma)
                self.model_params_update = tf.train.AdamOptimizer(self.model_learning_rate).minimize(self.model_loss)
            elif self.learn_env_model:
                self.model_n_h1 = params['model_n_h1']
                self.model_n_h2 = params['model_n_h2']
                self.model_s_input, self.model_a_input, self.model_sp_diff, self.query_r, self.query_gamma \
                    = create_nobinary_continuous_action_model('envmodel', self.stateDim, self.actionDim,
                                                     self.model_n_h1, self.model_n_h2)
                self.query_sp = self.model_sp_diff + self.model_s_input
                self.model_sp_diff_target, self.model_r_target, self.model_gamma_target, self.model_loss \
                    = model_loss('model_loss', self.stateDim, self.model_sp_diff, self.query_r, self.query_gamma)
                self.model_params_update = tf.train.AdamOptimizer(self.model_learning_rate).minimize(self.model_loss)

    def init_vars_stats(self, trainvars):
        for tvar in trainvars:
            self.n_total_vars += int(np.prod(tvar.shape))
            self.shapelist.append(tvar.shape)
            self.sizelist.append(int(np.prod(tvar.shape)))

    def update_fisher_statistics(self, grad_s):
        self.mu_grad_s, self.fisher_s_first, self.fisher_s \
                = update_stats(grad_s, self.mu_grad_s, self.fisher_s_first, self.count)
        self.count += 1

    def compute_action_value(self, scopename):
        raise NotImplementedError

    def define_loss(self, scopename):
        raise NotImplementedError

    def train(self, s, a, sp, r, gamma):
        raise NotImplementedError

    def take_action(self, state):
        raise NotImplementedError

    def train_env_model(self, s, a, sp, r, gamma, diff_weight = None):
        self.sess.run(self.model_params_update, feed_dict={self.model_s_input: s, self.model_a_input: a,
                                                           self.model_sp_diff_target: sp - s,
                                                               self.model_r_target: r, self.model_gamma_target: gamma})
        return None


    def get_reward(self, s, a, sp):
        return None


    def model_query(self, s, a):
        if len(s.shape) < 2:
            s = s[None, :]
            a = np.array([a])
        sp, r, g = self.sess.run([self.query_sp, self.query_r, self.query_gamma],
                                  feed_dict={self.model_s_input: s, self.model_a_input: a})
        return sp, np.squeeze(r), np.squeeze(g)


    def save_model(self, env_name, agent_name):
        with self.g.as_default():
            nt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            name = env_name + agent_name + nt
            filepath = name + '/nnmodel.ckpt'
            if not os.path.exists(name):
                os.makedirs(name)
            '''init saver'''
            savepath = self.saver.save(self.sess, filepath)
            print('model is saved at %s ' % savepath)


    def restore(self, file_path):
        tf.reset_default_graph()
        '''init saver'''
        self.saver.restore(self.sess, file_path)
        print('model restored !!!!!!')
