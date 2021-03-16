from collections import deque
import random
import numpy as np
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


class RecencyBuffer(object):

    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        # Right side of deque contains newest experience
        self.buffer = deque(maxlen=self.buffer_size)

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

    def clear(self):
        self.buffer.clear()