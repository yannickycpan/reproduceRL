import tensorflow as tf
import numpy as np


'''' ------------------ usage example: construct a two-layer QNN with FTA on the second layer -------------------- '''
''' SparseActFunc is an instantce of the class FTA , the FTA configuration is my suggestion '''

def create_fta_qnn(*args):

    scopename, dtype, n_input, n_output, n_hidden1, n_hidden2, unittype, SparseActFunc = args
    with tf.variable_scope(scopename):
        state_input = tf.placeholder(dtype, [None, n_input])
        # hidden1 = tf.layers.dense(state_input, n_hidden1, activation=SparseActFunc.func)
        hidden1 = tf.layers.dense(state_input, n_hidden1, activation=tf.nn.relu)
        sparse_phi = tf.layers.dense(hidden1, n_hidden2, activation=SparseActFunc.func)
        SparseActFunc.set_extra_act_strength(hidden1, n_hidden2)

        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        q_values = tf.layers.dense(sparse_phi, n_output, activation=None, kernel_initializer=w_init)

        max_qvalue = tf.reduce_max(q_values, axis=1)
        max_ind = tf.argmax(q_values, axis=1)

        # get variables
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return state_input, q_values, max_qvalue, max_ind, sparse_phi, tvars


class FTAConfiguration(object):
    default_attributes = {'n_tiles': 20, 'n_tilings': 1, 'sparse_dim': None,
                          'fta_input_max': 20.0, 'fta_input_min': -20.0, 'fta_eta': 2.0,
                          'outofbound_reg': 0.0, 'extra_strength': False,
                          'individual_tiling': False,
                          'actfunctypeFTA': 'linear',  'actfunctypeFTAstrength': 'linear'}

    def __init__(self, configdict):
        for key in configdict:
            if key in self.default_attributes:
                setattr(self, key, configdict[key])
                
        if not hasattr(self, 'fta_input_min'):
            self.fta_input_min = -self.fta_input_max
        if not hasattr(self, 'fta_eta'):
            self.fta_eta = (self.fta_input_max - self.fta_input_min)/self.n_tiles
            
        for key in self.default_attributes:
            if not hasattr(self, key):
                setattr(self, key, self.default_attributes[key])
        if self.n_tilings > 1:
            ''' if multi-tiling, use default setting, fta_input_max should be a list '''
            return


class FTA(object):

    act_func_dict = {'tanh': tf.nn.tanh, 'linear': lambda x: x,
                     'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid, 'clip': None, 'sin': tf.math.sin}

    def __init__(self, params):
        config = FTAConfiguration(params)
        self.config = config
        ''' rewrite the clip activation '''
        self.act_func_dict['clip'] = lambda x: tf.clip_by_value(x, config.fta_input_min, config.fta_input_max)

        self.extra_strength = config.extra_strength
        ''' set up sparsity control eta variable '''
        self.fta_eta = config.fta_eta
        ''' set up activation function before FTA '''
        self.actfunctypeFTA = config.actfunctypeFTA
        ''' set up activation function used for self-strength '''
        self.actfunctypeFTAstrength = config.actfunctypeFTAstrength
        ''' set up FTA bound '''
        self.outofbound_reg = config.outofbound_reg
        self.fta_loss = 0.0
        self.extra_act_strength = 1.

        self.n_tiles = config.n_tiles
        self.n_tilings = config.n_tilings
        self.individual_tiling = config.individual_tiling

        ''' NOTE: when using individual tiling, number of tilings must be equal to FTA's input dimension '''

        ''' set tilings, tiles '''
        if self.config.n_tilings > 1:
            self.c_mat, self.tile_delta_vector \
                = self.get_multi_tilings(config.n_tilings, config.n_tiles)
        else:
            self.c_vec, self.tile_delta, self.tiling_low_bound, self.tiling_up_bound \
                = self.get_tilings(config.n_tilings, config.n_tiles, config.fta_input_min, config.fta_input_max)

        if 'RBF' in params['name']:
            self.func = self.RBF_func
        elif self.config.n_tilings > 1 and not self.individual_tiling:
            self.func = self.FTA_func_multi_tiling
        elif self.config.n_tilings > 1 and self.individual_tiling:
            self.func = self.FTA_func_individual_tiling
        else:
            self.func = self.FTA_func
        print(' fta_eta, n_tilings, and n_tiles :: ===================================================== ',
              self.fta_eta, self.n_tilings, self.n_tiles)

    def Iplus_eta(self, x, eta):
        if eta == 0:
            return tf.math.sign(x)
        return tf.cast(x <= eta, tf.float32) * x + tf.cast(x > eta, tf.float32)

    def _sum_relu(self, c, x, delta):
        return tf.nn.relu(c - x) + tf.nn.relu(x - delta - c)

    ''' low bound must be nonpositive and up bound must be nonneg '''

    def compute_out_of_bound_loss(self, input):
        if self.outofbound_reg > 0 and self.actfunctypeFTA not in ['tanh', 'sigmoid', 'clip']:
            self.fta_loss = tf.reduce_mean(tf.reduce_sum(tf.cast((input > self.tiling_up_bound), tf.float32)
                                                         * input, axis=1)) \
                            - tf.reduce_mean(tf.reduce_sum(tf.cast((input < self.tiling_low_bound), tf.float32)
                                                           * input, axis=1))
            self.fta_loss = self.outofbound_reg * self.fta_loss

    def set_extra_act_strength(self, input, n_h):
        if self.extra_strength:
            self.extra_act_strength = tf.contrib.layers.fully_connected(input, n_h,
                                    activation_fn=self.act_func_dict[self.actfunctypeFTAstrength])

    def _get_strength(self, x, d, c):
        if x is None:
            return 1.0
        if self.extra_strength:
            strength = tf.reshape(self.extra_act_strength, [-1, d, 1])
        else:
            strength = 1.0
        return strength

    ''' for each tiling, operates on all of the input units; if rawinput is None, strenght is one '''
    def get_sparse_vector(self, input, rawinput, n_tiles, tile_delta, fta_eta, c):
        d = int(input.shape.as_list()[1])
        k = int(n_tiles)
        x = tf.reshape(input, [-1, d, 1])
        onehot = tf.reshape((1.0 - self.Iplus_eta(self._sum_relu(c, x, tile_delta),
                                         fta_eta)) * self._get_strength(rawinput, d, c), [-1, d * k])
        return onehot

    ''' rawinput = previouslayeroutput W + b '''
    def FTA_func(self, rawinput):
        """ this activation function decides if we should preprocess before feeding into FTA function """
        input = self.act_func_dict[self.actfunctypeFTA](rawinput)
        self.compute_out_of_bound_loss(input)
        onehot = self.get_sparse_vector(input, rawinput, self.n_tiles, self.tile_delta, self.fta_eta, self.c_vec)
        print(' after FTA processing the onehot dimension is :: ', onehot.shape)
        return onehot

    ''' no need for out of boundary loss for RBF '''

    def RBF_func(self, input):
        input = self.act_func_dict[self.actfunctypeFTA](input)
        d = int(input.shape.as_list()[1])
        k = self.n_tiles
        x = tf.reshape(input, [-1, d, 1])
        onehot = tf.reshape(tf.exp(-tf.square(self.c_vec - x) / self.fta_eta), [-1, d * k])
        print(' after RBF processing the sparse dimension is :: ', onehot.shape)
        return onehot

    def get_tilings(self, n_tilings, n_tile, input_min, input_max):
        tile_delta = (input_max - input_min) / n_tile
        if n_tilings == 1:
            one_c = np.linspace(input_min, input_max, n_tile, endpoint=False).astype(np.float32)
            c_vec = tf.constant(one_c)
            tile_delta = one_c[1] - one_c[0]
            return c_vec, tile_delta, input_min, input_max
        maxoffset = n_tilings * (input_max - input_min) / n_tile
        tiling_length = input_max - input_min + maxoffset
        startc = input_min - np.random.uniform(0, maxoffset, n_tilings)
        c_list = []
        for n in range(n_tilings):
            one_c = np.linspace(startc[n], startc[n] + tiling_length, n_tile, endpoint=False).astype(np.float32)
            c_list.append(tf.constant(one_c.copy().astype(np.float32)))
        tiling_low_bound = np.min(startc) - maxoffset
        tiling_up_bound = np.max(startc) + tiling_length
        return c_list, tile_delta, tiling_low_bound, tiling_up_bound


    ''' 
    get multiple tiling vectors, now fta_input_max is a list of upper bounds;
    need to study what tilings should be used
    '''
    def get_multi_tilings(self, n_tilings, n_tile):
        input_max_list = np.random.choice(self.config.fta_input_max, n_tilings)
        c_list = []
        tile_defta_list = []
        for n in range(n_tilings):
            ind = n % len(input_max_list)
            one_c = np.linspace(-input_max_list[ind], input_max_list[ind], n_tile, endpoint=False).astype(np.float32)
            c_list.append(tf.constant(one_c.copy().astype(np.float32).reshape((-1, n_tile))))
            tile_defta_list.append((one_c[1]-one_c[0]))
        c_mat = tf.concat(c_list, axis=0)
        tile_defta_vector = tf.reshape(tf.constant(np.array(tile_defta_list).astype(np.float32)), [n_tilings, 1])
        return c_mat, tile_defta_vector


    ''' 
    rawinput has shape (minibatchsize, # of hidden units)
    for example, rawinput = h W, h is the previous layer's output,
    W is the weight matrix in the current hidden layer whose activation is FTA.
    If h is a-by-b, W is b-by-c, then rawinput is a-by-c. If FTA has n_tilings and n_tiles, 
    then the output has shape (a, c * n_tilings * n_tiles). 
    '''
    def FTA_func_multi_tiling(self, rawinput):
        input = self.act_func_dict[self.actfunctypeFTA](rawinput)
        d = int(input.shape.as_list()[1])
        x = tf.reshape(input, [-1, d, 1, 1])
        ''' each row in the c_mat is a tiling, for each sample's each hidden unit, apply all those tilings  '''
        onehots = 1.0 - self.Iplus_eta(self._sum_relu(self.c_mat, x, self.tile_delta_vector),
                                       self.tile_delta_vector)
        onehot = tf.reshape(onehots, [-1, int(d * self.n_tiles * self.n_tilings)])
        print(' after FTA processing the onehot dimension is :: ', onehot.shape)
        return onehot


    '''
    individualized tiling: each element in a vector uses its own tiling. Hence this is 
    different with FTA_func_multi_tiling, where each element goes through all tilings vectors
    '''
    def FTA_func_individual_tiling(self, rawinput):
        input = self.act_func_dict[self.actfunctypeFTA](rawinput)
        d = int(input.shape.as_list()[1])
        x = tf.reshape(input, [-1, d, 1])
        ''' each row in the c_mat is a tiling, for each sample's each hidden unit, apply one tiling  '''
        onehots = 1.0 - self.Iplus_eta(self._sum_relu(self.c_mat, x, self.tile_delta_vector),
                                       self.tile_delta_vector)
        onehot = tf.reshape(onehots, [-1, int(d * self.n_tiles)])
        print(' after FTA processing the onehot dimension is :: ', onehot.shape)
        return onehot
