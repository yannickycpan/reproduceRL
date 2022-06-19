import tensorflow as tf

def create_qnn_inputsparse(*args):
    scopename, dtype, n_input, n_output, n_hidden1, n_hidden2, unittype, SparseActFunc = args
    with tf.variable_scope(scopename):
        state_input = tf.placeholder(dtype, [None, n_input])
        sparse_state_input = SparseActFunc.func(state_input)
        print(' sparse input feature is used ============================== ')
        hidden1 = tf.layers.dense(sparse_state_input, n_hidden1, activation=unittype)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=unittype)

        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        q_values = tf.layers.dense(hidden2, n_output, activation=None, kernel_initializer=w_init)

        max_qvalue = tf.reduce_max(q_values, axis=1)
        max_ind = tf.argmax(q_values, axis=1)

        # get variables
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return state_input, q_values, max_qvalue, max_ind, sparse_state_input, tvars

def create_qnn(*args):
    scopename, dtype, n_input, n_output, n_hidden1, n_hidden2, unittype = args
    with tf.variable_scope(scopename):
        state_input = tf.placeholder(dtype, [None, n_input])
        hidden1 = tf.layers.dense(state_input, n_hidden1, activation=unittype)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=unittype)

        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        q_values = tf.layers.dense(hidden2, n_output, activation=None, kernel_initializer=w_init)

        max_qvalue = tf.reduce_max(q_values, axis=1)
        max_ind = tf.argmax(q_values, axis=1)
        # get variables
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return state_input, q_values, max_qvalue, max_ind, tvars


def create_qnn_sparsebyreg(*args):
    scopename, dtype, n_input, n_output, n_hidden1, n_hidden2, unittype, regtype = args
    with tf.variable_scope(scopename):
        state_input = tf.placeholder(dtype, [None, n_input])
        hidden1 = tf.layers.dense(state_input, n_hidden1, activation=unittype)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=unittype)

        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        q_values = tf.layers.dense(hidden2, n_output, activation=None, kernel_initializer=w_init)

        if regtype == 1:
            philoss = tf.reduce_mean(tf.reduce_sum(tf.abs(hidden2), axis=1))
        else:
            philoss = tf.reduce_mean(tf.reduce_sum(tf.square(hidden2), axis=1))

        max_qvalue = tf.reduce_max(q_values, axis=1)
        max_ind = tf.argmax(q_values, axis=1)
        # get variables
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return state_input, q_values, max_qvalue, max_ind, philoss, hidden2, tvars


def create_fta_qnn(*args):
    scopename, dtype, n_input, n_output, n_hidden1, n_hidden2, unittype, SparseActFunc = args
    with tf.variable_scope(scopename):
        state_input = tf.placeholder(dtype, [None, n_input])
        hidden1 = tf.layers.dense(state_input, n_hidden1, activation=unittype)
        # hidden1 = tf.layers.dense(state_input, n_hidden1, activation=SparseActFunc.func)
        sparse_phi = tf.layers.dense(hidden1, n_hidden2, activation=SparseActFunc.func)

        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        q_values = tf.layers.dense(sparse_phi, n_output, activation=None, kernel_initializer=w_init)

        max_qvalue = tf.reduce_max(q_values, axis=1)
        max_ind = tf.argmax(q_values, axis=1)

        # get variables
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return state_input, q_values, max_qvalue, max_ind, sparse_phi, tvars