import numpy as np
import random
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from environments.environments import Environment
from terminations import choose_model

class GridWorld(Environment):
    def __init__(self, env_params):
        self.name = env_params['name']
        self.EPISODE_STEPS_LIMIT = env_params['EpisodeSamples']
        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0
        random.seed(env_params['seed'])
        np.random.seed(env_params['seed'])
        self.modelinfo = choose_model(env_params)
        self.modelinfo.model = self.model
        #print self.changeTime, self.sparseReward,self.randomStarts
        self.stepSize = 0.05
        self.state = None
        #right top corner
        self.righttop = np.array([1., 1.])
        self.sparseReward = env_params['sparseReward']


        """ if tough then the wall is wider """
        if 'Tough' in self.name:
            print('tough gridworld used !!!!!!!!!! ')
            self.hole_xy = np.array([0.4, 0.6])
            self.hole_h = 0.02
            self.hole_w = self.stepSize * 4.0
        else:
            self.hole_xy = np.array([0.5, 0.5])
            self.hole_h = 0.1
            self.hole_w = self.stepSize * 2.0
        self.noise = 0.01
        self.terminationNumber = 0
        self.mode = 0
        self.stepcount = 0

        self.stateDim = 2
        self.actionDim = 4
        self.statehigh = np.array([1.0, 1.0])
        self.statelow = np.array([0., 0.])
        self.actionBound = None

    def reset(self):
        self.state = np.random.uniform(0, 0.05, 2)
        self.stepcount = 0
        return self.state

    def model(self, s, a):
        x, y = (s[0], s[1]) if s is not None else (self.state[0], self.state[1])
        noisex = np.random.normal(0, self.noise, 1)[0]
        noisey = np.random.normal(0, self.noise, 1)[0]
        #a = a if random.uniform(0, 1.0) < 0.9 else random.randint(0, 3)
        if a == 0:
            y = y + self.stepSize + noisey
        elif a == 1:
            x = x + self.stepSize + noisex
        elif a == 2:
            y = y - self.stepSize + noisey
        elif a == 3:
            x = x - self.stepSize + noisex
        if self.hitWall(x, y):
            if s is None:
                x = self.state[0]
                y = self.state[1]
            else:
                x, y = s[0], s[1]
        x = np.clip(x, 0., 1.)
        y = np.clip(y, 0., 1.)
        reward = self.rewardFunction(x, y)
        reward = reward + np.random.normal(0.0, self.reward_noise_sigma)
        gamma = 0. if self.goalFunction(x, y) else None
        return np.array([x, y]), reward, gamma

    def step(self, a):
        sp, reward, _ = self.model(None, a)
        self.stepcount += 1
        self.state = sp.copy()
        return self.state, reward, self.goalFunction(self.state[0], self.state[1]), {}

    def hitWall(self, x, y):
        if x >= self.hole_xy[0] and x <= self.hole_xy[0] + self.hole_w:
            if y >= self.hole_xy[1] - self.hole_h and y <= self.hole_xy[1]:
                return False
            else:
                return True
        return False

    def goalFunction(self, x, y):
        if x > self.righttop[0] - 0.05 and y > self.righttop[1] - 0.05:
            return True
        #elif self.stepcount == self.EPISODE_STEPS_LIMIT:
        #    return True
        return False

    def rewardFunction(self, x, y):
        if self.sparseReward:
            return float(self.goalFunction(x, y))
        reward = -1. if not self.goalFunction(x, y) else 0.
        return reward

    def numObservations(self):
        return 2

    def numActions(self):
        return 4

    def close(self):
        return None


class TwoGoalGridWorld(GridWorld):
    def __init__(self, env_params):
        super(TwoGoalGridWorld, self).__init__(env_params)
        self.hole_xy = np.array([0.5, 0.1])
        self.hole_h = 0.1

        self.hole_xy1 = np.array([0.7, 0.3])
        self.hole_h1 = 0.1

        self.hole_w = self.stepSize * 2.0
        self.noise = 0.001

    def reset(self):
        self.state = np.random.uniform(0, 0.05, 2)
        self.stepcount = 0
        return self.state

    def goal1(self, x, y):
        if x > self.righttop[0] - 0.05 and y > self.righttop[1] - 0.05:
            print(' Goal 1 reached ------------------------- plus 1 ')
            return True
        return False

    def goal2(self, x, y):
        if x <= 0.05 and y >= 0.95:
            print(' Goal 2 reached ------------------------- plus 0.5 ')
            return True
        return False

    def goalFunction(self, x, y):
        if self.goal1(x, y) or self.goal2(x, y):
            return True
        return False

    def hitWall(self, x, y):
        if x >= self.hole_xy[0] and x <= self.hole_xy[0] + self.hole_w:
            if y >= self.hole_xy[1] - self.hole_h and y <= self.hole_xy[1]:
                return False
            else:
                return True
        elif x >= self.hole_xy1[0] and x <= self.hole_xy1[0] + self.hole_w:
            if y >= self.hole_xy1[1] - self.hole_h1 and y <= self.hole_xy1[1]:
                return False
            else:
                return True
        return False

    def rewardFunction(self, x, y):
        #if self.sparseReward:
        #    return float(self.goalFunction(x, y))
        reward = 0.
        if self.goal1(x, y):
            reward = 1.0
        elif self.goal2(x, y):
            reward = 0.5
        #elif self.hole_xy[0] < x < self.hole_xy[0] + 0.1 and y < self.hole_xy[1]:
        #    reward = -0.02
        return reward


class MazeGridWorld(Environment):
    def __init__(self, env_params):
        self.name = env_params['name']
        random.seed(env_params['seed'])
        np.random.seed(env_params['seed'])
        if 'Continuous' in self.name:
            self.model = self.model_conti
            self.stateDim = 2
            self.actionDim = 2
            self.statehigh = np.array([1.0, 1.0])
            self.statelow = np.array([0., 0.])
            self.actionBound = 1.
        else:
            self.model = self.model_disc
            self.stateDim = 2
            self.actionDim = 4
            self.statehigh = np.array([1.0, 1.0])
            self.statelow = np.array([0., 0.])
            self.actionBound = None
        self.EPISODE_STEPS_LIMIT = env_params['EpisodeSamples']
        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0

        self.modelinfo = choose_model(env_params)
        self.modelinfo.model = self.model
        #print self.changeTime, self.sparseReward,self.randomStarts
        self.stepSize = 0.05
        self.state = None
        #right top corner
        self.righttop = np.array([1., 1.])
        self.sparseReward = env_params['sparseReward'] if 'sparseReward' in env_params else False

        """ if tough then the wall is wider """
        # define x width
        self.hole_xy = np.array([0.2, 0.5])
        self.hole_h = 0.1
        self.hole_w = self.stepSize * 2.0

        self.hole_xy1 = np.array([0.4, 1.0])
        self.hole_xy2 = np.array([0.7, 0.2])

        self.noise = 0.01
        self.terminationNumber = 0
        self.mode = 0
        self.stepcount = 0

    def reset(self):
        self.state = np.random.uniform(0, 0.05, 2)
        self.stepcount = 0
        return self.state

    def model_disc(self, s, a):
        x, y = (s[0], s[1]) if s is not None else (self.state[0], self.state[1])
        noisex = np.random.normal(0, self.noise, 1)[0]
        noisey = np.random.normal(0, self.noise, 1)[0]
        #a = a if random.uniform(0, 1.0) < 0.9 else random.randint(0, 3)
        if a == 0:
            y = y + self.stepSize + noisey
        elif a == 1:
            x = x + self.stepSize + noisex
        elif a == 2:
            y = y - self.stepSize + noisey
        elif a == 3:
            x = x - self.stepSize + noisex
        if self.hitWall(x, y):
            if s is None:
                x = self.state[0]
                y = self.state[1]
            else:
                x, y = s[0], s[1]
        x = np.clip(x, 0., 1.)
        y = np.clip(y, 0., 1.)
        reward = self.rewardFunction(x, y)
        reward = reward + np.random.normal(0.0, self.reward_noise_sigma)
        gamma = 0. if self.goalFunction(x, y) else None
        return np.array([x, y]), reward, gamma

    def model_conti(self, s, a):
        x, y = (s[0], s[1]) if s is not None else (self.state[0], self.state[1])
        a = np.clip(a, -self.actionBound, self.actionBound)
        x = x + a[0]*self.stepSize
        y = y + a[1]*self.stepSize

        if self.hitWall(x, y):
            if s is None:
                x = self.state[0]
                y = self.state[1]
            else:
                x, y = s[0], s[1]
        x = np.clip(x, 0., 1.)
        y = np.clip(y, 0., 1.)
        reward = self.rewardFunction(x, y)
        gamma = 0. if self.goalFunction(x, y) else None
        return np.array([x, y]), reward, gamma

    def step(self, a):
        sp, reward, _ = self.model(None, a)
        self.stepcount += 1
        self.state = sp.copy()
        return self.state, reward, self.goalFunction(self.state[0], self.state[1]), {}

    def hitWall(self, x, y):
        if self.hole_xy[0] <= x <= self.hole_xy[0] + self.hole_w:
            if self.hole_xy[1] - self.hole_h <= y <= self.hole_xy[1]:
                return False
            else:
                return True
        if self.hole_xy1[0] <= x <= self.hole_xy1[0] + self.hole_w:
            if self.hole_xy1[1] - self.hole_h <= y <= self.hole_xy1[1]:
                return False
            else:
                return True
        if self.hole_xy2[0] <= x <= self.hole_xy2[0] + self.hole_w:
            if self.hole_xy2[1] - self.hole_h <= y <= self.hole_xy2[1]:
                return False
            else:
                return True
        return False

    def goalFunction(self, x, y):
        if x > self.righttop[0] - 0.05 and y > self.righttop[1] - 0.05:
            return True
        #elif self.stepcount == self.EPISODE_STEPS_LIMIT:
        #    return True
        return False

    def rewardFunction(self, x, y):
        if self.sparseReward:
            return float(self.goalFunction(x, y))
        reward = -1. if not self.goalFunction(x, y) else 0.
        return reward

    def numObservations(self):
        return 2

    def numActions(self):
        return 4

    def close(self):
        return None

class GridWorldContinuous(Environment):
    def __init__(self, env_params):
        self.name = env_params['name']
        self.EPISODE_STEPS_LIMIT = env_params['EpisodeSamples']
        random.seed(env_params['seed'])
        np.random.seed(env_params['seed'])
        self.modelinfo = choose_model(env_params)
        self.modelinfo.model = self.model
        #print self.changeTime, self.sparseReward,self.randomStarts
        self.stepSize = 0.05
        self.state = None
        #right top corner
        self.righttop = np.array([1., 1.])
        self.sparseReward = env_params['sparseReward']
        #use hole position, two holes, the second one indicates shorter path
        #(x1,y1, x2,y2)
        self.hole_xy = np.array([0.5, 0.5])
        self.hole_h = 0.1
        self.hole_w = self.stepSize*2.0
        self.noise = 0.01
        self.terminationNumber = 0
        self.mode = 0
        self.stepcount = 0

        self.stateDim = 2
        self.actionDim = 2
        self.statehigh = np.array([1.0, 1.0])
        self.statelow = np.array([0., 0.])
        self.actionBound = 1.

    def reset(self):
        self.state = np.random.uniform(0, 0.05, 2)
        self.stepcount = 0
        return self.state

    def model(self, s, a):
        x, y = (s[0], s[1]) if s is not None else (self.state[0], self.state[1])
        #noisex = np.random.normal(0, self.noise, 1)[0]
        #noisey = np.random.normal(0, self.noise, 1)[0]

        x = x + a[0]*0.05
        y = y + a[1]*0.05

        if self.hitWall(x, y):
            if s is None:
                x = self.state[0]
                y = self.state[1]
            else:
                x, y = s[0], s[1]
        x = np.clip(x, 0., 1.)
        y = np.clip(y, 0., 1.)
        reward = self.rewardFunction(x, y)
        gamma = 0. if self.goalFunction(x, y) else None
        return np.array([x, y]), reward, gamma

    def step(self, a):
        sp, reward, _ = self.model(None, a)
        self.stepcount += 1
        self.state = sp.copy()
        return self.state, reward, self.goalFunction(self.state[0], self.state[1]), {}

    def hitWall(self, x, y):
        if x >= self.hole_xy[0] and x <= self.hole_xy[0] + self.hole_w:
            if y >= self.hole_xy[1] - self.hole_h and y <= self.hole_xy[1]:
                return False
            else:
                return True
        return False

    def goalFunction(self, x, y):
        if x > self.righttop[0] - 0.05 and y > self.righttop[1] - 0.05:
            return True
        #elif self.stepcount == self.EPISODE_STEPS_LIMIT:
        #    return True
        return False

    def rewardFunction(self, x, y):
        if self.sparseReward:
            return float(self.goalFunction(x, y))
        reward = -1. if not self.goalFunction(x, y) else 0.
        return reward

    def numObservations(self):
        return 2

    def numActions(self):
        return 4

    def close(self):
        return None

class RiverSwim(Environment):
    def __init__(self, env_params):
        self.name = env_params['name']
        self.EPISODE_STEPS_LIMIT = env_params['EpisodeSamples']
        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0

        random.seed(env_params['seed'])
        np.random.seed(env_params['seed'])
        self.modelinfo = choose_model(env_params)
        self.modelinfo.model = self.model

        self.pos = 0.0
        self.stepSize = env_params['EnvStepSize'] if 'EnvStepSize' in env_params else 0.1
        self.noise = self.stepSize / 5.0

        self.stateDim = 1
        self.actionDim = 2
        self.statehigh = np.array([1.0])
        self.statelow = np.array([0.0])
        self.actionBound = None

    def getObs(self):
        return np.array([self.pos])

    def reset(self):
        self.n = 0
        self.pos = np.random.uniform(0, 0.05)
        return self.getObs()

    def model(self, s, a):
        x = self.pos if s is None else s[0]
        stepSize = self.stepSize + random.uniform(-self.noise, self.noise)
        if a == 0:
            x = x - stepSize
        elif a == 1:
            if np.random.uniform(0, 1.) < 0.6:
                x = x + stepSize
            else:
                x = x - stepSize
        x = np.clip(x, 0., 1.)
        gamma = None
        return np.array([x]), self.rewardFunction(x), gamma

    def step(self, a):
        sp, reward, gamma = self.model(None, a)
        self.pos = sp[0]
        return self.getObs(), reward, False, {}

    def rewardFunction(self, x):
        if 1.0 - x < self.stepSize/2.0:
            return 1.0
        if x < self.stepSize/2.0:
            return 5.0/1000.0
        return 0.0

    def numObservations(self):
        return 1

    def numActions(self):
        return 2

    def close(self):
        return None