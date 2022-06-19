import tensorflow as tf

''' continuous control version, create_actor_nn and create_critic_nn are for DDPG '''
def create_actor_nn(scopename, n_input, n_output, bound, n_hidden1=400, n_hidden2=300, usetanh=0):
    with tf.variable_scope(scopename):
        hidden_type = tf.nn.relu
        if usetanh == 1:
            hidden_type = tf.nn.tanh
        actor_input = tf.placeholder(tf.float32, [None, n_input])
        # normalizedactorinput = slim.batch_norm(actor_input, is_training = self.is_training)
        #hidden1 = tf.contrib.layers.fully_connected(actor_input, n_hidden1, activation_fn=tf.nn.relu)
        #hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, activation_fn=tf.nn.relu)
        hidden1 = tf.layers.dense(actor_input, n_hidden1, activation=hidden_type)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=hidden_type)
        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        #action = bound*tf.contrib.layers.fully_connected(hidden2, n_output, activation_fn=tf.nn.tanh,
        #                              weights_initializer=w_init)
        action = bound * tf.layers.dense(hidden2, n_output, activation=tf.nn.tanh,
                                                           kernel_initializer=w_init)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return actor_input, action, tvars


def create_critic_nn(*args):
    scopename, n_input, n_output, actor_output, n_hidden1, n_hidden2, usetanh = args
    with tf.variable_scope(scopename):
        hidden_type = tf.nn.relu
        if usetanh == 1:
            hidden_type = tf.nn.tanh
        state_input = tf.placeholder(tf.float32, [None, n_input])
        state_hidden1 = tf.layers.dense(state_input, n_hidden1, activation=hidden_type)
        #print('tanh hidden is used for critic ')
        # action directly go to second hidden layer
        action_input = actor_output
        hidden1 = tf.concat([state_hidden1, action_input], axis=1)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=hidden_type)
        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        #value = tf.contrib.layers.fully_connected(state_action_hidden2, n_output, activation_fn=None, weights_initializer=w_init)
        value = tf.layers.dense(hidden2, n_output, activation=None, kernel_initializer=w_init)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return state_input, action_input, value, tvars


def create_critic_nn_sparsebyreg(*args):
    scopename, n_input, n_output, actor_output, n_hidden1, n_hidden2, usetanh, regtype = args
    with tf.variable_scope(scopename):
        hidden_type = tf.nn.relu
        if usetanh == 1:
            hidden_type = tf.nn.tanh
        state_input = tf.placeholder(tf.float32, [None, n_input])
        state_hidden1 = tf.layers.dense(state_input, n_hidden1, activation=hidden_type)
        # action directly go to second hidden layer
        action_input = actor_output
        hidden1 = tf.concat([state_hidden1, action_input], axis=1)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=hidden_type)

        if regtype == 1:
            philoss = tf.reduce_mean(tf.reduce_sum(tf.abs(hidden2), axis=1))
        else:
            philoss = tf.reduce_mean(tf.reduce_sum(tf.square(hidden2), axis=1))

        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        value = tf.layers.dense(hidden2, n_output, activation=None, kernel_initializer=w_init)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return state_input, action_input, value, philoss, hidden2, tvars


def create_critic_fta_nn_vanilla(*args):
    scopename, n_input, n_output, actor_output, n_hidden1, n_hidden2, usetanh, SparseActFunc = args
    hidden_type = tf.nn.relu
    if usetanh == 1:
        hidden_type = tf.nn.tanh
    with tf.variable_scope(scopename):
        state_input = tf.placeholder(tf.float32, [None, n_input])
        action_input = actor_output

        hidden1 = tf.layers.dense(state_input, n_hidden1, activation=hidden_type)
        # SparseActFunc.set_extra_act_strength(hidden1, n_hidden2)
        sparse_phi = tf.layers.dense(tf.concat([hidden1, action_input], axis=1),
                                     n_hidden2, activation=SparseActFunc.func)

        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        value = tf.layers.dense(sparse_phi, n_output, activation=None, kernel_initializer=w_init)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return state_input, action_input, value, sparse_phi, tvars


def create_actor_fta_nn(*args):
    scopename, n_input, n_output, bound, n_hidden1, n_hidden2, usetanh, SparseActFunc = args
    hidden_type = tf.nn.relu
    if usetanh == 1:
        hidden_type = tf.nn.tanh
    with tf.variable_scope(scopename):
        actor_input = tf.placeholder(tf.float32, [None, n_input])

        hidden1 = tf.layers.dense(actor_input, n_hidden1, activation=hidden_type)
        # SparseActFunc.set_extra_act_strength(hidden1, n_hidden2)
        sparse_phi = tf.layers.dense(hidden1, n_hidden2, activation=SparseActFunc.func)

        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        action = bound * tf.layers.dense(sparse_phi, n_output, activation=tf.nn.tanh, kernel_initializer=w_init)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return actor_input, action, sparse_phi, tvars


def get_hidden_concat(state_input, action_input, n_hidden1, n_hidden2, hidden_type):
    state_hidden1 = tf.layers.dense(state_input, n_hidden1, activation=hidden_type)
    # action directly go to second hidden layer
    state_action_input = tf.concat([state_hidden1, action_input], axis=1)
    hidden2 = tf.layers.dense(state_action_input, n_hidden2, activation=hidden_type)
    return hidden2


def get_hidden_dot(state_input, action_input, n_hidden1, n_hidden2, hidden_type):
    state_hidden1 = tf.layers.dense(state_input, n_hidden1, activation=hidden_type)
    action_hidden1 = tf.layers.dense(action_input, n_hidden1, activation=hidden_type)
    # action directly go to second hidden layer
    state_action_input = state_hidden1 * action_hidden1
    hidden2 = tf.layers.dense(state_action_input, n_hidden2, activation=hidden_type)
    return hidden2