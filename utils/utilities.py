import numpy as np

''' wishart matrix '''
def get_psd_square_matrix(dim):
    # limit = np.sqrt(3./dim)
    u = np.random.normal(0, 1, [dim, dim])
    S = u.dot(u.T)
    w, _ = np.linalg.eig(S)
    return 1./np.max(w) * S

def get_skew_symmetric_matrix(dim):
    psd = get_psd_square_matrix(dim)
    for i in range(dim):
        psd[i, i] = 0.
        psd[i, i:] = -psd[i, i:]
    return psd

def get_orthogonal_matrix(dim):
    psd = get_psd_square_matrix(dim)
    _, v = np.linalg.eig(psd)
    return v

def get_tridiag_matrix(dim):
    u = get_psd_square_matrix(dim)
    for i in range(dim):
        for j in range(dim):
            if abs(i - j) >= 2:
                u[i, j] = 0.
    return u

''' get upper triang mat '''
def get_triang_matrix(dim):
    psd = get_psd_square_matrix(dim)
    return np.triu(psd)

def sherman_update(Kinv, u, v):
    u = u.reshape((len(u), 1))
    v = v.reshape((len(v), 1))
    Kinvdotu = np.dot(Kinv, u)
    #print(np.inner(v, Kinvdotu))
    denominator = 1.0 + np.dot(v.T, Kinvdotu)[0, 0]

    vdotKinv = np.dot(v.T, Kinv)
    newKinv = Kinv - np.outer(1.0/denominator * Kinvdotu, vdotKinv)
    return newKinv

def SMInv(Ainv, u, v, e = None):
    u = u.reshape((len(u),1))
    v = v.reshape((len(v),1))
    if e is not None:
        g = np.dot(Ainv, u) / (e + np.dot(v.T, np.dot(Ainv, u)))
        return (Ainv / e) - np.dot(g, np.dot(v.T, Ainv/e))
    else:
        return Ainv - np.dot(Ainv, np.dot(np.dot(u, v.T), Ainv)) / ( 1 + np.dot(v.T, np.dot(Ainv, u)))

'''given Kinv, compute (alpha*K + beta * uv^T)^inv '''
def sherman_update_coeff(Kinv, u, v, alpha, beta):
    u = u.reshape((len(u), 1))
    v = v.reshape((len(v), 1))

    Kinvdotu = Kinv.dot(u)
    #innerprod = max(np.inner(Kinvdotu, v), 0.0)
    denominator = alpha**2 + alpha*beta*np.dot(v.T, Kinvdotu)[0, 0]
    vdotKinv = np.dot(v.T, Kinv)
    #if np.inner(Kinvdotu, v) < 0:
    #    print(u, v, np.inner(Kinvdotu, v))
    #print(Kinvdotu.shape, vdotKinv.shape, beta/denominator, np.inner(Kinvdotu, v))
    newKinv = (1.0/alpha) * Kinv - np.outer((beta/denominator) * Kinvdotu, vdotKinv)
    return newKinv

def convert2onehot(nparr, dim):
    onehot = np.zeros((nparr.shape[0], dim))
    onehot[np.arange(nparr.shape[0]), nparr] = 1.0
    return onehot


#compute the discounted sum of vec
def discount(vec, rate):
    import scipy.signal
    assert vec.ndim >= 1
    return scipy.signal.lfilter([1], [1, -rate], vec[::-1], axis=0)[::-1]


def discount_returns(gamma, rewards):
    discounted_returns = np.zeros_like(rewards)
    running_add = 0
    for t in range(len(rewards)-1, -1, -1):
        running_add = running_add * gamma + rewards[t]
        discounted_returns[t] = running_add
    return discounted_returns


def multi_trajs_returns(gammas, rewards):
    inds = np.where(np.array(gammas) == 0)[0]
    discounted_returns = np.zeros_like(rewards)
    start = 0
    for i in range(inds.shape[0]):
        discounted_returns[start:(inds[i]+1)] = discount_returns(gammas[0], rewards[start:(inds[i]+1)])
        start = inds[i]+1
    return discounted_returns