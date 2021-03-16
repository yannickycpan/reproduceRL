import numpy as np

smallConst = 1e-5

def weight_init(n_prev, n_next):
    winit = np.random.normal(np.linspace(-1.0 / n_prev, 1.0 / n_prev, n_next).reshape((n_next, 1)),
                             1.0 / np.sqrt(n_prev),
                             (n_next, n_prev)).astype(np.float32)
    print(' the mean value of each column is ', np.mean(winit.T, axis=0))
    return winit.T

def compute_activation_overlap(phix, phixp):
    act_overlap = np.mean(np.sum((np.abs(phix) > smallConst) * (np.abs(phixp) > smallConst), axis=1))
    return act_overlap


def compute_instance_sparsity(phix):
    sparse_prop = np.mean(np.sum((np.abs(phix) > smallConst), axis=1))
    return sparse_prop


def compute_activation_freq(phi):
    return np.sum(phi > 0, axis=0)/float(phi.shape[0])


def compute_sparse_dim(np_c):
    return int(np.sum([arr.shape[0] for arr in np_c]))

def tfget_sparsities(sess, stateinput, phinode, replaybuffer, batchSize, statescale):
    state, _, _, _, _ = replaybuffer.sample_batch(batchSize)
    sphi = sess.run(phinode, feed_dict={stateinput: state / statescale})
    overlapsparsity = compute_activation_overlap(sphi[:int(batchSize/2), :], sphi[-int(batchSize/2):, :])
    instancesparsity = compute_instance_sparsity(sphi)
    return overlapsparsity, instancesparsity

def tfget_sparsities_state_action_pair(sess, stateinput, actioninput, phinode, replaybuffer, batchSize, statescale):
    state, action, _, _, _ = replaybuffer.sample_batch(batchSize)
    sphi = sess.run(phinode, feed_dict={stateinput: state / statescale, actioninput: action})
    overlapsparsity = compute_activation_overlap(sphi[:int(batchSize/2), :], sphi[-int(batchSize/2):, :])
    instancesparsity = compute_instance_sparsity(sphi)
    return overlapsparsity, instancesparsity


def compute_batch_dot(vec1, vec2):
    normvec1 = np.linalg.norm(vec1, ord=2, axis=1) + 1e-5
    normvec2 = np.linalg.norm(vec2, ord=2, axis=1) + 1e-5

    vec1, vec2 = vec1 / normvec1[:, None], vec2 / normvec2[:, None]

    vecprod = vec1 * vec2
    innerprods = np.sum(vecprod, axis=1)
    avgdotprod = np.mean(innerprods)
    avgneginnerprods = np.mean(innerprods[innerprods < 0]) if np.sum(innerprods < 0) > 0 else 0.
    propneg = np.sum(innerprods < 0)/vecprod.shape[0]
    return avgdotprod, avgneginnerprods, propneg


def get_grad_interferences(gradvecs, suffix=''):
    batchSize = gradvecs.shape[0]
    vec1 = gradvecs[:int(batchSize / 2), :]
    vec2 = gradvecs[int(batchSize / 2):, :]
    instancesp = np.mean(np.sum(np.abs(gradvecs) > 0.00001, axis=1))
    avgdotprod, avgneginnerprods, propneg = compute_batch_dot(vec1, vec2)
    interferencedict = {'GradDot'+suffix: avgdotprod, 'NegGradDot'+suffix: avgneginnerprods,
                'NegGradDotProp'+suffix: propneg, 'InstanceGradSparse'+suffix: instancesp}
    return interferencedict