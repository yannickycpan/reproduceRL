import tensorflow as tf

def get_w_b(scopename, n_input, n_hidden1, n_hidden2, add_dim=0):
    W1 = tf.get_variable(scopename + "/h1/weights", [n_input, n_hidden1])  # get weight in hidden layer
    b1 = tf.get_variable(scopename + "/h1/biases", [n_hidden1], initializer=tf.zeros_initializer())  # get weight in hidden layer

    W2 = tf.get_variable(scopename + "/h2/weights", [n_hidden1+add_dim, n_hidden2])  # get weight in output layer
    b2 = tf.get_variable(scopename + "/h2/biases", [n_hidden2], initializer=tf.zeros_initializer())  # get bias in output layer
    return W1, b1, W2, b2


def get_2layer_output(input, W1, b1, W2, b2, act_fn):
    hidden1 = act_fn(tf.matmul(input, W1) + b1)
    hidden2 = act_fn(tf.matmul(hidden1, W2) + b2)
    return hidden2


def get_2layer_output_additional_input(state_input, action_input, W1, b1, W2, b2, act_fn):
    hidden1 = act_fn(tf.matmul(state_input, W1) + b1)
    concathidden1 = tf.concat([hidden1, action_input], axis=1)
    hidden2 = act_fn(tf.matmul(concathidden1, W2) + b2)
    return hidden2


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv2d_lta(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return x


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def get_w_b_atari():
    weights = {
        'wc1': tf.get_variable('W0', shape=(8, 8, 4, 32)),
        'wc2': tf.get_variable('W1', shape=(4, 4, 32, 64)),
        'wc3': tf.get_variable('W2', shape=(3, 3, 64, 64)),
        'wd1': tf.get_variable('W3', shape=(3136, 256)),
        #'wd2': tf.get_variable('W4', shape=(3136, 128)),
        #'out': tf.get_variable('W6', shape=(512, n_output)),
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(32)),
        'bc2': tf.get_variable('B1', shape=(64)),
        'bc3': tf.get_variable('B2', shape=(64)),
        'bd1': tf.get_variable('B3', shape=(256)),
        #'bd2': tf.get_variable('B4', shape=(128)),
        #'out': tf.get_variable('Bout', shape=(n_output)),
    }
    return weights, biases


def atari_conv_net(x, weights, biases):
    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], 4)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    # conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 2)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    # conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 1)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    # conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    #print('fc1 shape is ================================= ', fc1.shape)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output action values
    # finally we multiply the fully connected layer with the weights and add a bias term.
    # out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return fc1

def get_w_b_minatari(nchannels, n_h1):
    ks = 3
    num_filters = 16
    def size_linear_unit(size, kernel_size=ks, stride=1):
        return (size - (kernel_size - 1) - 1) // stride + 1
    linear_input_size = size_linear_unit(10) * size_linear_unit(10) * num_filters
    weights = {
        'wc1': tf.get_variable('W0', shape=(ks, ks, nchannels, num_filters)),
        'wd1': tf.get_variable('W1', shape=(linear_input_size, n_h1))
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(num_filters)),
        'bd1': tf.get_variable('B1', shape=(n_h1))
    }
    return weights, biases


def minatari_conv_net(x, weights, biases):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    fc1 = tf.layers.flatten(conv1)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    return fc1


def get_w_b_minatari_largernn(nchannels, n_h1, n_h2):
    ks = 5
    num_filters = 32
    def size_linear_unit(size, kernel_size=ks, stride=1):
        return (size - (kernel_size - 1) - 1) // stride + 1
    linear_input_size = size_linear_unit(10) * size_linear_unit(10) * num_filters
    weights = {
        'wc1': tf.get_variable('W0', shape=(ks, ks, nchannels, num_filters)),
        'wd1': tf.get_variable('W1', shape=(linear_input_size, n_h1)),
        'wd2': tf.get_variable('W2', shape=(n_h1, n_h2))
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(num_filters)),
        'bd1': tf.get_variable('B1', shape=(n_h1)),
        'bd2': tf.get_variable('B2', shape=(n_h2))
    }
    return weights, biases


def minatari_conv_largernet(x, weights, biases):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    fc1 = tf.layers.flatten(conv1)
    print(' fc1 size is ===================================', fc1.shape)
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1']))
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2']))
    return fc2

def get_min_atari_phi_and_weights(x, nchannels, n_h1, n_h2, xsp=None):
    if n_h2 == 0:
        weights, biases = get_w_b_minatari(nchannels, n_h1)
        hidden = minatari_conv_net(x, weights, biases)
        sphidden = None if xsp is None else minatari_conv_net(xsp, weights, biases)
    else:
        weights, biases = get_w_b_minatari_largernn(nchannels, n_h1, n_h2)
        hidden = minatari_conv_largernet(x, weights, biases)
        sphidden = None if xsp is None else minatari_conv_largernet(xsp, weights, biases)
    return hidden, sphidden
