import tensorflow as tf
import sys
import os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
# from operations import extract_3d_tensor, extracted_2d_tensor, model_loss
# from customizednn import get_w_b, get_2layer_output


def create_discrete_action_model(scopename, stateDim, actionDim, n_hidden1, n_hidden2):
    with tf.variable_scope(scopename):
        model_state_input = tf.placeholder(tf.float32, [None, stateDim])
        hidden1 = tf.layers.dense(model_state_input, n_hidden1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
        nextStates_diff =  tf.layers.dense(hidden2, actionDim*stateDim, activation=None)
        nextStates_diff = tf.reshape(nextStates_diff, [-1, actionDim, stateDim])

        #rhidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2)
        nextRewards =  tf.layers.dense(hidden2, actionDim, activation=None)

        #ghidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2)
        nextGammas =  tf.layers.dense(hidden2, actionDim, activation=None)
        # nextGammas = tf.log(tf.exp(nextGammas))
        # get variables
        # tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return model_state_input, nextStates_diff, nextRewards, nextGammas


def create_continuous_action_model(scopename, stateDim, actionDim, n_binary, n_hidden1, n_hidden2):
    with tf.variable_scope(scopename):
        model_state_input = tf.placeholder(tf.float32, [None, stateDim])
        model_action_input = tf.placeholder(tf.float32, [None, actionDim])
        hidden1_state = tf.contrib.layers.fully_connected(model_state_input, n_hidden1)
        hidden1_action = tf.contrib.layers.fully_connected(model_action_input, n_hidden1)

        hidden1 = tf.concat([hidden1_state, hidden1_action], axis=1)
        hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2)

        nextStates_nb = tf.contrib.layers.fully_connected(hidden2, stateDim - n_binary, activation_fn=None)
        nextStates_b = tf.contrib.layers.fully_connected(hidden2, n_binary, activation_fn=None)
        nextStates_b_labels = tf.cast(tf.nn.sigmoid(nextStates_b) > 0.5, tf.float32)

        hidden2_rewards = tf.contrib.layers.fully_connected(hidden1, n_hidden2)
        nextRewards = tf.contrib.layers.fully_connected(hidden2_rewards, 1, activation_fn=None)
    return model_state_input, model_action_input, nextStates_nb, nextStates_b, nextStates_b_labels, nextRewards


def create_nobinary_continuous_action_model(scopename, stateDim, actionDim, n_hidden1, n_hidden2):
    with tf.variable_scope(scopename):
        model_state_input = tf.placeholder(tf.float32, [None, stateDim])
        model_action_input = tf.placeholder(tf.float32, [None, actionDim])

        hidden1 = tf.layers.dense(tf.concat([model_state_input, model_action_input], axis=1), n_hidden1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)

        nextStates = tf.layers.dense(hidden2, stateDim, activation=None)
        nextRewards = tf.layers.dense(hidden2, 1, activation=None)
        nextGammas = tf.layers.dense(hidden2, 1, activation=None)
    return model_state_input, model_action_input, nextStates, nextRewards, nextGammas

def create_nobinary_continuous_action_state_model(scopename, stateDim, actionDim, n_hidden1, n_hidden2):
    with tf.variable_scope(scopename):
        model_state_input = tf.placeholder(tf.float32, [None, stateDim])
        model_action_input = tf.placeholder(tf.float32, [None, actionDim])
        hidden1_state = tf.contrib.layers.fully_connected(model_state_input, n_hidden1)
        hidden1_action = tf.contrib.layers.fully_connected(model_action_input, n_hidden1)

        hidden1 = tf.concat([hidden1_state, hidden1_action], axis=1)
        hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2)

        nextStates = tf.contrib.layers.fully_connected(hidden2, stateDim, activation_fn=None)
    return model_state_input, model_action_input, nextStates