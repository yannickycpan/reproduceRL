import numpy as np

TDHC = 'TDHC'
VHC = 'ValueHC'
MIXTWO = 'MixedSum'
MIXGRAD = 'MixedGrad'
MIXHESS = 'MixedHess'
PROPMIX = 'ProportionSum'

smallConst = 0.00001
scalarCovMat = 0.0001

''' Roboschool are not in use '''
def projection(env_name, ss, low, up):
    s = np.clip(ss, low, up)
    if env_name == 'Acrobot-v1':
        s[:2] = s[:2] / np.linalg.norm(s[:2], ord=2)
        s[2:4] = s[2:4] / np.linalg.norm(s[2:4], ord=2)
    elif env_name == "RoboschoolReacher-v1":
        s[4:6] = s[4:6] / np.linalg.norm(s[4:6], ord=2)
    elif env_name in ["RoboschoolInvertedPendulum-v1", "RoboschoolInvertedPendulumSwingup-v1"]:
        s[2:4] = s[2:4] / np.linalg.norm(s[2:4], ord=2)
    elif "Roboschool" in env_name:
        s[1:3] = s[1:3] / np.linalg.norm(s[1:3], ord=2)
    elif env_name == "Reacher-v2":
        norm1 = np.sqrt(s[0] ** 2 + s[2] ** 2)
        norm2 = np.sqrt(s[1] ** 2 + s[3] ** 2)
        s[0], s[2] = s[0] / norm1, s[2] / norm1
        s[1], s[3] = s[1] / norm2, s[3] / norm2
    #if self.n_binary > 0:
    #    s[-self.n_binary:] = 1.0 * (s[-self.n_binary:] > 0.5)
    return s


def projected_noisy_natural_gradient(env_name, noise_scale, ga_learning_rate, deltas, covmat, s, low, up):
    ng_deltas = covmat.dot(deltas)
    norm_ng_deltas = np.linalg.norm(ng_deltas, ord=2)
    normalized_ng_deltas = ng_deltas / norm_ng_deltas \
        if (norm_ng_deltas > smallConst and not np.isnan(norm_ng_deltas)) else np.zeros_like(ng_deltas)
    noise = noise_scale * np.random.normal(np.zeros_like(s), np.sqrt(np.diagonal(covmat)), s.shape)
    s_updated = s + ga_learning_rate * normalized_ng_deltas + noise
    s_updated = projection(env_name, s_updated, low, up)
    return s_updated, norm_ng_deltas, np.linalg.norm(s_updated - s, ord=2), deltas, normalized_ng_deltas


def get_mixedqueue_init(hctype, erbuff, scqueue):
    if scqueue[VHC].getSize() == 0 or hctype == VHC:
        init_s, _, _, _, _ = erbuff.sample_batch(1)
        hctype = VHC
    else:
        init_s = scqueue[VHC].sample_batch(1)
    return hctype, init_s

def get_init_state_hc(env_name, noise_scale, erbuff, scqueue, low, up, covmat, hctype, algo_name):
    if 'MixedQueue' in algo_name or 'MixedSepQueue' in algo_name \
            or 'MixedGradQueue' in algo_name or 'MixedHessQueue' in algo_name:
        hctype, init_s = get_mixedqueue_init(hctype, erbuff, scqueue)
    else:
        init_s, _, _, _, _ = erbuff.sample_batch(1)
    init_s = init_s[0, :]
    noise = np.random.normal(np.zeros_like(init_s), np.sqrt(np.diagonal(covmat)), init_s.shape) * noise_scale
    s = projection(env_name, init_s + noise, low, up)
    return s, hctype

""" time steps is considered as the number of samples in ER buffer """

def get_hctype(algo_name, name2q):
    if 'MixedQueue' in algo_name or 'MixedSepQueue' in algo_name:
        hctype = np.random.choice([VHC, MIXTWO], 1)[0]
        return hctype
    elif 'MixedGradQueue' in algo_name:
        hctype = np.random.choice([VHC, MIXGRAD], 1)[0]
        return hctype
    elif 'MixedHessQueue' in algo_name:
        hctype = np.random.choice([VHC, MIXHESS], 1)[0]
        return hctype
    hctype = name2q[algo_name]
    return hctype

''' --------------------------hill climbing search control-------------------- '''

'''
    A NOTE: in general, tanh is better than relu; on GridWorld domain, 
    tanh gradually gives zero gradient, while relu always gives zero;
    on MountainCar, tanh gives non-zero gradient for a relatively long time.
    relu always gives zero. 
'''


def backup_termination(s):
    return False


def sdist(s, ss):
    return np.linalg.norm(s - ss, ord=2) / np.sqrt(np.squeeze(s).shape[0])


def ga_hc_search_control(erbuff, scqueue, low, up, thres, covmat, model, check_termination, agent_func, name2q, prob=None):

    algo_name, env_name, update_count, noise_scale, hc_update, num_sc_samples, maxgaloops, ga_learning_rate \
        = agent_func.name, agent_func.env_name, agent_func.update_count, agent_func.noise_scale, \
          agent_func.hc_update, agent_func.num_sc_samples, agent_func.maxgaloops, agent_func.ga_learning_rate

    if check_termination is None:
        check_termination = backup_termination

    # find search-control type
    hctype = get_hctype(algo_name, name2q)
    n_added = 0
    nloop = 0
    restarttimes = 0
    s, hctype = get_init_state_hc(env_name, noise_scale, erbuff, scqueue, low, up, covmat, hctype, algo_name)
    lastadded = s

    ''' num of gradient steps actually means number of search control states'''
    while n_added < num_sc_samples:
        nloop += 1
        ''' s_delta means how much the update moved '''
        deltas = hc_update(s, model, hctype, prob)
        s, norm_ng_deltas, s_delta, g, ng \
            = projected_noisy_natural_gradient(env_name, noise_scale,
                                               ga_learning_rate, deltas, covmat, s, low, up) \
                                                if hctype != 'Rollout' else deltas

        movedist = sdist(s, lastadded)
        ''' if all state vars are within the boundary, put it into sc queue '''
        if movedist >= thres and np.all((s >= low) * (s <= up)) and not check_termination(s):
            # and (agent_func.useTrueModel is False and sdist(s, inits) <= 4.0 * thres):
            lastadded = s
            if 'MixedQueue' in algo_name:
                scqueue[VHC].add(lastadded.copy())
            else:
                scqueue[hctype].add(lastadded.copy())
            n_added += 1
        ''' if outside of boundary restart hc '''
        if (np.all((s < low) + (s > up)) or check_termination(s) or s_delta < smallConst)\
                and n_added < num_sc_samples:
            s, hctype = get_init_state_hc(env_name, noise_scale, erbuff, scqueue, low, up, covmat, hctype, algo_name)
            hctype = get_hctype(algo_name, name2q)
            lastadded = s
            restarttimes += 1
        if nloop > maxgaloops:
            break
    ''' the algo return number of added samples per step '''
    return n_added/nloop