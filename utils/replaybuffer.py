from collections import deque
import random
import numpy as np
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from SumTree import SumTree
from utilities import convert2onehot, discount

class RecencyBuffer(object):

    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        # Right side of deque contains newest experience
        self.buffer = deque(maxlen = self.buffer_size)

    def add(self, s, a, sp, r, gamma, additional = None):
        if additional is None:
            self.buffer.append([s, a, sp, r, gamma])
        else:
            self.buffer.append([s, a, sp, r, gamma, additional])

    def getSize(self):
        return len(self.buffer)

    def get_batch_actions(self, inds):
        batch = [self.buffer[i] for i in inds]
        s, a, sp, r, g = map(np.array, zip(*batch))
        return a

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    ''' when call this fun, suppose buffer size is the same as k '''
    def sample_recent_k(self, k):
        batch = [self.buffer[i] for i in range(k)]
        return map(np.array, zip(*batch))

    def sample_batch_seq_discrete_action(self, batch_size, plan_depth, actionDim):
        batch_ints = np.random.randint(0, self.getSize() - plan_depth, batch_size).tolist()
        batchsamples = []
        for ind in batch_ints:
            #print(ind)
            cursample = [self.buffer[ind+i] for i in range(plan_depth)]
            s, a, sp, r, gamma, behavepi = map(np.array, zip(*cursample))
            actionseq = convert2onehot(a, actionDim)
            planlen_ind = np.where(gamma == 0.0)[0]
            if len(planlen_ind) == 0:
                # terminal state is not inside
                # planlen_ind = plan_depth - 1
                #planlen_ind =
                planlen_ind = np.random.randint(plan_depth)
            else:
                planlen_ind = planlen_ind[0]
                #print(' batch include termination !!! planlen id is :: ', planlen_ind)
            discounted_sumr = np.sum([r[i]*(gamma[i]**i) for i in range(planlen_ind + 1)])
            poweredgamma = gamma[planlen_ind]**(planlen_ind+1.)
            batchsamples.append([s[0, :], actionseq, sp[planlen_ind, :], discounted_sumr,
                                 poweredgamma, np.prod(behavepi[:planlen_ind+1]), planlen_ind+1])
        return map(np.array, zip(*batchsamples))

    def save_to_file(self):
        s, a, sp, r, gamma = map(np.array, zip(*self.buffer))
        np.savetxt('trains.txt', s, fmt='%10.7f', delimiter=',')
        np.savetxt('traina.txt', a, fmt='%10.7f', delimiter=',')
        np.savetxt('trainsp.txt', sp, fmt='%10.7f', delimiter=',')
        np.savetxt('trainr.txt', r, fmt='%10.7f', delimiter=',')
        np.savetxt('traing.txt', gamma, fmt='%10.7f', delimiter=',')

    def clear(self):
        self.buffer.clear()


class RecencyStateBuffer(RecencyBuffer):
    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        # Right side of deque contains newest experience
        self.buffer = None
        self.count = 0
        self.similarity = None
        self.uniform_ints = np.arange(0, self.buffer_size, 10)

    def add(self, s, a = None, sp = None, r = None, gamma = None):
        if self.buffer is None:
            self.buffer = np.zeros((self.buffer_size, s.shape[0]))
        self.buffer[int(self.count % self.buffer_size), :] = s
        self.count += 1
        return True

    def similarity_add_covmat(self, s, covmat_inv):
        diff = s - self.buffer[self.uniform_ints, :]
        diff = np.sum(diff.dot(covmat_inv) * diff, axis=1)
        #if np.sum(sims[self.uniform_ints]) > self.uniform_ints.shape[0]/2.0:
        #    return
        #if np.min(np.sum(np.abs(diff), axis=1)) < 0.01:
        if np.min(diff) < 0.001:
            #print('diff and min dis are:: ', diff, np.min(np.sum(np.abs(diff), axis=1)))
            return False
        self.buffer[int(self.count % self.buffer_size), :] = s
        self.count += 1
        return True

    def similarity_add(self, s):
        if self.buffer is None:
            self.buffer = np.zeros((self.buffer_size, s.shape[0]))
        diff = s - self.buffer[self.uniform_ints, :]
        if np.min(np.sum(np.abs(diff), axis=1)) < 0.00001:
            #print('diff and min dis are:: ', diff, np.min(np.sum(np.abs(diff), axis=1)))
            return False
        self.buffer[int(self.count % self.buffer_size), :] = s
        self.count += 1
        return True

    def getSize(self):
        return min(self.count, self.buffer_size)

    def sample_batch(self, batch_size):
        batch_ints = np.random.randint(0, min(self.count, self.buffer_size), batch_size)
        return self.buffer[batch_ints,:]

class StateBuffer(object):

    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        # Right side of deque contains newest experience
        self.buffer = deque(maxlen = self.buffer_size)

    def add(self, s):
        self.buffer.append(s[None, :])

    def getSize(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return np.vstack(batch)

    def clear(self):
        self.buffer.clear()


class StateActionBuffer(object):

    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.buffer = deque(maxlen = self.buffer_size)

    def add(self, s, a):
        self.buffer.append([np.squeeze(s), np.squeeze(a)])

    def getSize(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def clear(self):
        self.buffer.clear()


""" 
PrioritizedER comes from 
https://github.com/takoika/PrioritizedExperienceReplay
"""

class PrioritizedER(object):
    """ The class represents prioritized experience replay buffer.
    The class has functions: store samples, pick samples with
    probability in proportion to sample's priority, update
    each sample's priority, reset alpha.
    see https://arxiv.org/pdf/1511.05952.pdf .
    """
    beta = 0.0

    def __init__(self, memory_size, alpha):
        """ Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        # self.batch_size = batch_size
        self.alpha = alpha

    def add(self, priority, data):
        """ Add new sample.

        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, priority ** self.alpha)

    def sample_batch(self, batch_size):
        """ The method return samples randomly.

        Parameters
        ----------
        beta : float

        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        if self.tree.filled_size() < batch_size:
            return None, None, None

        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1. / self.memory_size / priority) ** self.beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0])  # To avoid duplicating

        self.priority_update(indices, priorities)  # Revert priorities

        weights /= max(weights)  # Normalize for stability

        pbs, pba, pbsp, pbr, pbgamma = map(np.array, zip(*out))

        return pbs, pba, pbsp, pbr, pbgamma, indices, weights

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p ** self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.
        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)


#each time use all samples in the buffer and then clean the buffer
#there is no buffer size but batchsize = number of episodes we want to store
class TrajectoryBuffer(object):
    def __init__(self, batchSize, gamma):
        self.batchSize = batchSize
        self.gamma = gamma
        self.buffer = None
        self.curSize = 0
        self.n_cur = 0
        self.buffer = []
        for i in range(self.batchSize):
            self.buffer.append(self.default_dict())

    def default_dict(self):
        return {'states':[], 'nextstates':[], 'actions':[], 'rewards':[], 'gammas': [], 'terminated': False}

    def add(self, s, a, sp, r, episodeEnd):
        self.buffer[self.n_cur]['states'].append(s)
        self.buffer[self.n_cur]['nextstates'].append(sp)
        self.buffer[self.n_cur]['actions'].append(a)
        self.buffer[self.n_cur]['rewards'].append(r)
        if episodeEnd:
            self.buffer[self.n_cur]['gammas'].append(0.0)
            self.buffer[self.n_cur]['terminated'] = True
            self.n_cur = self.n_cur + 1
        else:
            self.buffer[self.n_cur]['gammas'].append(self.gamma)
        return None

    def get_one_batch(self, batchindex):
        if batchindex > len(self.buffer)-1:
            print('Batch does not exist')
            return None
        traj = self.buffer[batchindex]
        temp = self.default_dict()
        temp['states'] = np.array(traj['states'])
        temp['nextstates'] = np.array(traj['nextstates'])
        temp['actions'] = np.array(traj['actions'])
        temp['rewards'] = np.array(traj['rewards'])
        return temp

    def get_samples(self):
        samples = []
        for traj in self.buffer:
            temp = self.default_dict()
            temp['states'] = np.array(traj['states'])
            temp['nextstates'] = np.array(traj['nextstates'])
            temp['actions'] = np.array(traj['actions'])
            temp['rewards'] = np.array(traj['rewards'])
            temp['gammas'] = np.array(traj['gammas'])
            samples.append(temp)
        self.compute_advantages(samples)
        concated_trajs = self.concat_trajs(samples)
        return concated_trajs

    def compute_advantages(self, trajs):
        for traj in trajs:
            traj['returns'] = discount(traj['rewards'], self.gamma)
            traj['advantages'] = traj['returns']

    # NOTE: advantage normalization is in the PG.py file
    def concat_trajs(self, trajs):
        newdict = {}
        newdict['states'] = np.concatenate([traj['states'] for traj in trajs])
        newdict['rewards'] = np.concatenate([traj['rewards'] for traj in trajs])
        newdict['gammas'] = np.concatenate([traj['gammas'] for traj in trajs])
        newdict['nextstates'] = np.concatenate([traj['nextstates'] for traj in trajs])
        newdict['actions'] = np.concatenate([traj['actions'] for traj in trajs])
        newdict['returns'] = np.concatenate([traj['returns'] for traj in trajs])
        return newdict

    def reset(self):
        for traj in self.buffer:
            for key in traj:
                if key != 'terminated':
                    del traj[key][:]
            traj['terminated'] = False
        self.n_cur = 0