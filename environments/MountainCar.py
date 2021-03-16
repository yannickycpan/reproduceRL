import numpy as np
import math
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from modelinfo import ModelInfo

class MountainCar(object):
    def __init__(self, env_params):
        self.stateBounded = env_params['stateBounded']
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.observation_high = np.array([self.max_position, self.max_speed])
        self.observation_low = np.array([self.min_position, -self.max_speed])
        self.goal_position = 0.5

        self.modelinfo = ModelInfo
        self.modelinfo.termination_condition = self.termination_mcar \
            if not self.stateBounded else self.termination_mcar_bounded
        self.modelinfo.model = self.model

        self.EPISODE_STEPS_LIMIT = env_params['EpisodeSamples']
        self.env_name = env_params['name']
        self.sparseReward = env_params['sparseReward'] if 'sparseReward' in env_params else False
        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0
        print('reward sigma is :: ------------------------ ', self.reward_noise_sigma)
        self.act_noise = env_params['actNoise'] if 'actNoise' in env_params else 0.3
        self.state_noise = env_params['stateNoise'] if 'stateNoise' in env_params else 0.1

        self.state = None

        self.stateDim = 2
        self.actionDim = 3

        self.timestep = 0
        if self.stateBounded:
            self.statehigh = np.array([1.0, 1.0])
            self.statelow = np.array([0., 0.])
        else:
            self.statehigh = np.array([0.6, 0.07])
            self.statelow = np.array([-1.2, -0.07])
        self.actionBound = None

    def reset(self):
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0.])
        self.timestep = 0
        return self.state.copy()

    def add_obs_noise(self, position, velocity):
        position = position + np.random.normal(0.0, self.state_noise)
        velocity = velocity + np.random.normal(0.0, self.state_noise/10.)
        return position, velocity

    def model(self, s, action):
        #if 'Rand' in self.env_name:
        #    action = np.random.randint(self.actionDim) if np.random.uniform(0., 1.) < self.act_noise else action
        if s is None:
            position, velocity = self.state[0], self.state[1]
        else:
            position, velocity = s[0], s[1]
            if self.stateBounded:
                position = position*(self.max_position-self.min_position) + self.min_position
                velocity = velocity*(self.max_speed*2.) - self.max_speed
        velocity += (action - 1) * 0.001 + math.cos(3 * position) * (-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0): velocity = 0
        done = bool(position >= self.goal_position)
        reward = -1.0 if not self.sparseReward else 0
        if 'Rand' in self.env_name:
            reward = reward + np.random.normal(0.0, self.reward_noise_sigma)
        gamma = 0. if done else None
        reward = 1. if done else reward
        if self.stateBounded and s is not None:
            position = (position - self.min_position)/(self.max_position - self.min_position)
            velocity = (velocity + self.max_speed)/(2.*self.max_speed)
            return np.array([position, velocity]), reward, gamma
        return np.array([position, velocity]), reward, gamma

    def termination_mcar(self, s):
        if len(s.shape) == 1:
            return True if s[0] > 0.5 else False
        dones = (s[:, 0] > 0.5)
        return dones

    def termination_mcar_bounded(self, s):
        if len(s.shape) == 1:
            position = s[0]
            position = position * (self.max_position - self.min_position) + self.min_position
            return True if position > 0.5 else False
        positions = s[:, 0] * (self.max_position - self.min_position) + self.min_position
        dones = (positions > 0.5)
        return dones

    def step(self, a):
        sp, reward, gamma = self.model(None, a)
        self.state = sp.copy()
        self.timestep += 1
        done = (bool(gamma == 0.) or self.timestep == self.EPISODE_STEPS_LIMIT)
        if self.stateBounded:
            return (self.state - self.observation_low)/(self.observation_high - self.observation_low), reward, bool(gamma == 0.), None
        return self.state, reward, done, None

    def close(self):
        return None

class MountainCarContinuous(object):
    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = 0.0015

        #self.low_state = np.array([self.min_position, -self.max_speed])
        #self.high_state = np.array([self.max_position, self.max_speed])
        self.statehigh = np.array([0.6, 0.07])
        self.statelow = np.array([-1.2, -0.07])

        self.reset()

    def model(self, s, action):
        if s is None:
            position, velocity = self.state[0], self.state[1]
        else:
            position, velocity = s[0], s[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)

        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action[0],2)*0.1

        gamma = 0. if done else None
        return np.array([position, velocity]), reward, gamma

    def step(self, a):
        sp, reward, gamma = self.model(None, a)
        self.state = sp.copy()
        return self.state, reward, bool(gamma == 0.), None

    def reset(self):
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0.])
        return self.state.copy()

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55


class MountainCarStop(MountainCar):
    def __init__(self, env_params):
        super(MountainCarStop, self).__init__(env_params)

        self.goal_position = 0.55
        self.goal_vel = 0.001

        self.modelinfo = ModelInfo
        self.modelinfo.termination_condition = self.termination_mcar
        self.modelinfo.model = self.model

        if self.stateBounded:
            self.statehigh = np.array([1.0, 1.0])
            self.statelow = np.array([0., 0.])
        else:
            self.statehigh = np.array([0.6, 0.07])
            self.statelow = np.array([-1.2, -0.07])
        self.actionBound = None

    def model(self, s, action):
        if s is None:
            position, velocity = self.state[0], self.state[1]
        else:
            position, velocity = s[0], s[1]
            if self.stateBounded:
                position = position * (self.max_position - self.min_position) + self.min_position
                velocity = velocity * (self.max_speed * 2.) - self.max_speed
        velocity += (action - 1) * 0.001 + math.cos(3 * position) * (-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0): velocity = 0
        done = bool(position >= self.goal_position)
        reward = self.rewardfunc(position, velocity)
        #reward = 1.0 if done else 0.0
        gamma = 0. if done else None
        if self.stateBounded and s is not None:
            position = (position - self.min_position) / (self.max_position - self.min_position)
            velocity = (velocity + self.max_speed) / (2. * self.max_speed)
            return np.array([position, velocity]), reward, gamma
        return np.array([position, velocity]), reward, gamma

    def rewardfunc(self, position, velocity):
        reward = -1
        if bool(position >= self.goal_position) and bool(-self.goal_vel <= velocity) and bool(velocity <= self.goal_vel):
            #print(' 100 reward got !!!!!!!!!!!! ')
            reward = 100.
        return reward

    def termination_mcar(self, s):
        if len(s.shape) == 1:
            return True if s[0] >= self.goal_position else False
        dones = (s[:, 0] > self.goal_position)
        return dones

    def step(self, a):
        sp, reward, gamma = self.model(None, a)
        self.state = sp.copy()
        self.timestep += 1
        done = (bool(gamma == 0.) or self.timestep == self.EPISODE_STEPS_LIMIT)
        if self.stateBounded:
            return (self.state - self.observation_low)/(self.observation_high - self.observation_low), reward, bool(gamma == 0.), None
        return self.state, reward, done, None

    def close(self):
        return None

    def reset(self):
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0.])
        self.timestep = 0
        return self.state.copy()