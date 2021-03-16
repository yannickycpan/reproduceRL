import tensorflow as tf
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

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

    def init_vars_stats(self, trainvars):
        return

    def update_fisher_statistics(self, grad_s):
        return

    def compute_action_value(self, scopename):
        raise NotImplementedError

    def define_loss(self, scopename):
        raise NotImplementedError

    def train(self, s, a, sp, r, gamma):
        raise NotImplementedError

    def take_action(self, state):
        raise NotImplementedError

    def train_env_model(self, s, a, sp, r, gamma, diff_weight = None):
        return


    def get_reward(self, s, a, sp):
        return


    def model_query(self, s, a):
        return

    def save_model(self, env_name, agent_name):
        return

    def restore(self, file_path):
        return
