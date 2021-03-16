import tensorflow as tf
import math


def create_logpi_gaussian(act_mu, act_sigma, action_holders, actionDim):
    const_term = actionDim * tf.log(2. * math.pi)
    sigma_logdet = tf.reduce_sum(tf.log(tf.sqrt(act_sigma)), axis=1)
    action_diff = act_mu - action_holders
    exp_term = tf.reduce_sum(tf.divide(tf.square(action_diff), act_sigma), axis=1)
    logpi_dist = -0.5 * (const_term + sigma_logdet + exp_term)
    pi_dist = tf.exp(logpi_dist)
    neg_log_prob = -logpi_dist
    entropy = 0.5 * sigma_logdet
    return pi_dist, neg_log_prob, entropy


def decor_loss(phi, scalor=1./64.):
    phibar = tf.reduce_mean(phi, axis=0, keepdims=True)
    covMat = scalor * tf.matmul(phi - phibar, phi - phibar, transpose_a=True)
    diagkey = tf.diag(tf.matrix_diag_part(covMat))
    offdiag = covMat - diagkey
    return tf.reduce_mean(tf.square(offdiag))


''' input is a 3D tensor: ?-by-dim-by-dim matrix,  
    for each matrix I generate a single random vector,
     one can also generate multiple rand vec for each '''


def neg_sample_pd_loss(A, batchSize, dim):
    randx = tf.random.normal(shape=(batchSize, dim, 1), mean=0., stddev=0.1)
    loss = tf.reduce_mean(tf.nn.relu(-tf.reduce_sum(randx * tf.matmul(A, randx), axis=1)))
    return loss


def neg_sample_pd_loss_2d(A, n, dim):
    randx = tf.random.normal(shape=(n, dim), mean=0., stddev=0.1)
    loss = tf.reduce_mean(tf.nn.relu(-tf.reduce_sum(randx * tf.matmul(randx, A), axis=1)))
    return loss

def orth_reg_loss(F, dim):
    return tf.reduce_mean(tf.reduce_sum(tf.square(tf.matmul(F, F, transpose_b=True) - tf.eye(dim)), axis=[0, 1]))

def rotation_reg_loss(F, dim):
    return orth_reg_loss(F, dim) + tf.reduce_mean(tf.square(tf.linalg.det(F) - 1.))

def skew_sym_loss(F):
    return tf.reduce_mean(tf.square(F + tf.transpose(F)))

def upper_triang_loss(F):
    return tf.reduce_mean(tf.square(tf.matrix_band_part(F, -1, 0)))

def tridiag_loss(F):
    return tf.reduce_mean(tf.square(tf.matrix_band_part(F, 1, 1)))

def update_reg_critic_ops(critic_output, xAy, Anorm_loss, lr, beta=1.0, eta=1.0, emphatic_weights=None):
    critic_value_holders = tf.placeholder(tf.float32, [None])
    if emphatic_weights is None:
        vanillaloss = tf.reduce_mean(tf.square(critic_value_holders - tf.squeeze(critic_output)))
    else:
        vanillaloss = tf.reduce_mean(tf.square(critic_value_holders - tf.squeeze(critic_output))*emphatic_weights)
    psd_loss = tf.reduce_mean(tf.abs(tf.minimum(xAy, 0.0)))# - tf.reduce_mean(tf.minimum(xAy, 0.001) * tf.cast(xAy >= 0, tf.float32))
    print('psd reg and Anorm reg are respectively :: ----------------------------- ', beta, eta)
    critic_loss = vanillaloss + beta * psd_loss + eta * Anorm_loss
    critic_update = tf.train.AdamOptimizer(lr).minimize(critic_loss)
    return critic_value_holders, critic_update, psd_loss, vanillaloss, Anorm_loss


def update_critic_ops(critic_output, lr, additional_loss1=0.0, regweight1=0.0, additional_loss2=0.0, regweight2=0.0):
    critic_target_holders = tf.placeholder(tf.float32, [None], name='critic_target_holders')
    critic_loss = tf.reduce_mean(tf.square(critic_target_holders - tf.squeeze(critic_output))) \
                  + regweight1 * additional_loss1 + regweight2 * additional_loss2
    critic_update = tf.train.AdamOptimizer(lr, name='critic_opt').minimize(critic_loss)
    return critic_target_holders, critic_update


def update_critic_cubic_ops(critic_output, lr, additional_loss=0.0, regweight=0.0):
    critic_target_holders = tf.placeholder(tf.float32, [None], name='critic_target_holders')
    critic_loss = tf.reduce_mean(tf.math.pow(tf.abs(critic_target_holders - tf.squeeze(critic_output)), 3)) \
                  + regweight * additional_loss
    critic_update = tf.train.AdamOptimizer(lr, name='critic_opt').minimize(critic_loss)
    print(' cubic objective used ----------------------------------------- ')
    return critic_target_holders, critic_update


''' input is list and output is also list '''
def create_vec_hess_prod(obj, list_vec, list_tvars):
    list_grad = tf.gradients(obj, list_tvars)
    grad_vec_inner_prod = tf.add_n([tf.reduce_sum(grad * vec)
                                   for (grad, vec) in zip(list_grad, list_vec)])
    vec_hess_prod = tf.gradients(grad_vec_inner_prod, list_tvars)
    return vec_hess_prod


''' input is flattened list and output is also flattened list (vector) '''
def create_vec_hess_prod_flatten(obj, vec, list_tvars, sizelist, shapelist):
    list_vec = reshape2vars(vec, sizelist, shapelist)
    list_grad = tf.gradients(obj, list_tvars)
    grad_vec_inner_prod = tf.add_n([tf.reduce_sum(grad * vec)
                                   for (grad, vec) in zip(list_grad, list_vec)])
    vec_hess_prod = tf.gradients(grad_vec_inner_prod, list_tvars)
    vec_hess_prod_flatten = listshape2vec(vec_hess_prod)
    return vec_hess_prod_flatten


def listshape2vec(list_tvars):
    return tf.concat([tf.reshape(varpart, [-1]) if varpart is not None else tf.reshape(0., [-1])
                      for varpart in list_tvars], axis=0)


def reshape2vars(vectorholder, sizelist, shapelist):
    shapedholders = []
    count = 0
    for idx in range(len(shapelist)):
        shapedholders.append(tf.reshape(vectorholder[count:count + sizelist[idx]], shapelist[idx]))
        count += sizelist[idx]
    return shapedholders


''' this is used in HC Dyna '''
def compute_normsquare_gradient_hessian(scopename, stateDim, maxq_grad_s, state_input, max_q):
    with tf.variable_scope(scopename):
        grad_norm_square = tf.reduce_sum(tf.square(maxq_grad_s), axis=1)
        normed_grad_s = tf.gradients(grad_norm_square, [state_input])[0]

        hessian = tf.stack(
            [tf.reshape(tf.gradients(maxq_grad_s[:, idx], [state_input])[0], [-1])
             for idx in range(stateDim)], axis=0)
        hessian_norm_square = tf.reduce_sum(tf.square(hessian))
        hess_norm_grad_s = tf.gradients(hessian_norm_square, [state_input])[0]

        print('test shapes are :: ', grad_norm_square.shape, max_q.shape)
        product_grad_s = tf.gradients((grad_norm_square + hessian_norm_square) * max_q, [state_input])[0]

        return product_grad_s, normed_grad_s, hess_norm_grad_s


def huber_loss(mat, kaap):
    condition = tf.cast((tf.abs(mat) <= kaap), tf.float32)
    huberloss_mat = 0.5 * tf.square(mat) * condition + kaap * (tf.abs(mat) - 0.5 * kaap) * (1.0 - condition)
    return huberloss_mat


def quantile_loss(scopename, numQuantiles, kaap, q_quantiles, tau_hat):
    with tf.variable_scope(scopename):
        target_quantiles = tf.placeholder(tf.float32, [None, numQuantiles], name='q_holders')
        target_test = tf.reshape(target_quantiles, [-1, 1, numQuantiles])
        q_quantiles_test = tf.reshape(q_quantiles, [-1, numQuantiles, 1])
        '''u_mat is a b * N * N size tensor below commented line was my implementation'''
        u_mat = tf.reshape(target_quantiles, [-1, 1, numQuantiles]) \
                - tf.reshape(q_quantiles, [-1, numQuantiles, 1])
        huberloss_u_mat = huber_loss(u_mat, kaap)
        dirac_umat = tf.abs(tau_hat - tf.cast((u_mat < 0.0), tf.float32))
        final_loss_mat = dirac_umat * huberloss_u_mat
        final_loss = tf.reduce_sum(tf.reduce_mean(final_loss_mat, axis=2), axis=1)
        final_loss = tf.reduce_mean(final_loss)
        #final_loss = tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(final_loss_mat, axis=0), axis=1))
    return target_quantiles, final_loss, target_test, q_quantiles_test, u_mat


def compute_q_acted(scopename, action_input, actionDim, q_values, tar_q_values=None):
    action_one_hot = tf.one_hot(action_input, actionDim, 1.0, 0.0)
    q_acted = tf.reduce_sum(q_values * action_one_hot, axis=1)
    if tar_q_values is not None:
        tar_q_acted = tf.reduce_sum(tar_q_values * action_one_hot, axis=1)
    else:
        tar_q_acted = None
    return q_acted, tar_q_acted


'''extrat from a 3D tensor some certain line along 2nd axis'''
'''example usage: in quantile regression, the output has size b * k * n, we want each sample's best qvalue quantiles'''
'''input indexes must be 1d array, more efficient implementation???'''
def extract_3d_tensor(tensor, indexes):
    rowinds = tf.range(0, tf.cast(tf.shape(tensor)[0], tf.int64), 1)
    ind_nd = tf.concat([tf.reshape(rowinds, [-1, 1]), tf.reshape(indexes, [-1, 1])], axis=1)
    extracted = tf.gather_nd(tensor, ind_nd)
    return extracted

'''extract the indexes elements along the second dim of tensor'''
def extracted_2d_tensor(tensor, indexes):
    one_hot = tf.one_hot(indexes, tf.shape(tensor)[1], 1.0, 0.0)
    extracted = tf.reduce_sum(tensor * one_hot, axis=1)
    return extracted


def update_target_nn_move(tar_tvars, tvars, tau):
    target_params_update = [tf.assign_add(tar_tvars[idx], tau * (tvars[idx] - tar_tvars[idx]))
                                 for idx in range(len(tvars))]
    return target_params_update

def update_target_nn_assign(tar_tvars, tvars):
    target_params_update = [tf.assign(tar_tvars[idx],  tvars[idx]) for idx in range(len(tvars))]
    return target_params_update

'''model based operations'''
def model_loss(scopename, stateDim, next_state, next_reward, next_gammas=None, s_w=1.0, r_w=1.0, gamma_w=1.0):
    with tf.variable_scope(scopename):
        state_target = tf.placeholder(tf.float32, [None, stateDim], name='state')
        #state_weight = tf.placeholder(tf.float32, [1, stateDim], name='state_w')
        reward_target = tf.placeholder(tf.float32, [None], name = 'reward')
        gamma_target = tf.placeholder(tf.float32, [None], name='gamma')
        if next_gammas is not None:
            gamma_loss = tf.losses.mean_squared_error(tf.exp(gamma_target), tf.exp(tf.squeeze(next_gammas)))
            print(' gamma loss is used ======================================== ')
        else:
            gamma_loss = 0.
        total_loss = s_w * tf.reduce_mean(tf.reduce_sum(tf.square(state_target - next_state), axis = 1)) \
                     + r_w * tf.losses.mean_squared_error(reward_target, tf.squeeze(next_reward)) \
                     + gamma_w * gamma_loss
        return state_target, reward_target, gamma_target, total_loss

def feature_learning_loss(scopename, stateDim, next_state, next_reward, next_gamma, predicted_qvalue = None):
    with tf.variable_scope(scopename):
        state_target = tf.placeholder(tf.float32, [None, stateDim], name='state')
        reward_target = tf.placeholder(tf.float32, [None], name = 'reward')
        gamma_target = tf.placeholder(tf.float32, [None], name='gamma')
        actionvale_target = tf.placeholder(tf.float32, [None], name='qvalue')
        #qvalue_weight = tf.placeholder(tf.float32, shape=(), name="qweight")
        total_loss = tf.reduce_mean(tf.reduce_sum(tf.square(state_target - next_state), axis = 1)) \
                     + tf.losses.mean_squared_error(reward_target, tf.squeeze(next_reward)) \
                     + tf.losses.mean_squared_error(gamma_target, tf.squeeze(next_gamma))
                     #+ qvalue_weight * tf.losses.mean_squared_error(actionvale_target, tf.squeeze(predicted_qvalue))
        return state_target, reward_target, gamma_target, actionvale_target, total_loss

def state_model_loss(scopename, stateDim, next_state):
    with tf.variable_scope(scopename):
        state_target = tf.placeholder(tf.float32, [None, stateDim], name='state')
        state_weight = tf.placeholder(tf.float32, [1, stateDim], name='state_w')
        total_loss = tf.reduce_mean(tf.reduce_sum(tf.square(state_target - next_state)*state_weight, axis = 1))
        return state_target, state_weight, total_loss

def model_with_binary_loss(scopename, stateDim, n_binary, next_state_nb, next_state_b, next_reward):
    with tf.variable_scope(scopename):
        state_target_nb = tf.placeholder(tf.float32, [None, stateDim - n_binary], name='statenb')
        state_target_b = tf.placeholder(tf.float32, [None, n_binary], name='stateb')
        state_weight_nb = tf.placeholder(tf.float32, [1, stateDim - n_binary], name='state_w')
        reward_target = tf.placeholder(tf.float32, [None], name = 'reward')
        total_loss = tf.reduce_mean(tf.reduce_sum(tf.square(state_target_nb - next_state_nb)*state_weight_nb, axis=1)) \
                    + tf.losses.sigmoid_cross_entropy(state_target_b, next_state_b) \
                     + tf.losses.mean_squared_error(reward_target, tf.squeeze(next_reward))
        return state_target_nb, state_target_b, state_weight_nb, reward_target, total_loss