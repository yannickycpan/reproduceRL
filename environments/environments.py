import numpy as np
import random
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
try:
    import roboschool
except ImportError:
    print('Running without using roboschool')

from terminations import choose_model

#This file provide environments to interact with, consider actions as continuous, need to rewrite otherwise
class Environment(object):
    def __init__(self, env_params):
        import gym
        try:
            import highway_env
            # parking-v0 is a continuous control task
        except:
            print(' High way env is not available !!!!!!!!!!!!!!!! ')
        self.name = env_params['name']
        makename = self.name if not self.name.startswith('Disc') else self.name.replace('Disc', '')
        self.instance = gym.make(makename)
        self.instance.seed(env_params['seed'])
        #self.instance.seed = env_params['seed']
        random.seed(env_params['seed'])
        np.random.seed(env_params['seed'])
        # maximum number of steps allowed for each episode
        #self.TOTAL_STEPS_LIMIT = env_params['TotalSamples']
        if hasattr(self.instance, '_max_episode_steps'):
            self.EPISODE_STEPS_LIMIT = max(env_params['EpisodeSamples'], self.instance._max_episode_steps)
            print('-------------the original longest length is :: -----------------------', self.instance._max_episode_steps)
        else:
            self.EPISODE_STEPS_LIMIT = self.instance._max_episode_steps = env_params['EpisodeSamples']
        self.modelinfo = choose_model(env_params)

        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0
        self.actNoise = env_params['actNoise'] if 'actNoise' in env_params else 0.0

        #self.instance._max_episode_steps = env_params['EpisodeSamples']
        # state info
        self.stateDim = self.getStateDim()
        self.stateRange = self.getStateRange()
        self.stateMin = self.getStateMin()
        self.stateBounded = env_params['stateBounded']
        #self.stateBounded = False if np.any(np.isinf(self.instance.observation_space.high)) or np.any(np.isinf(self.instance.observation_space.low)) else True
        # action info
        self.actionDim = self.getControlDim()
        self.actionBound = self.getActBound()
        self.actMin = self.getActMin()

        #if self.name == 'Acrobot-v1':
        self.statehigh = self.instance.observation_space.high
        self.statelow = self.instance.observation_space.low
        if self.name == 'MountainCar-v0' and self.stateBounded:
            self.statehigh = np.array([1., 1.])
            self.statelow = np.array([0., 0.])

        print('stateDim:',self.stateDim)
        print("stateBounded :: ", self.stateBounded)
        print("actionDim", self.actionDim)
        print("actionBound :: ", self.actionBound)
        
    # Reset the environment for a new episode. return the initial state
    def reset(self):
        state = self.instance.reset()
        if self.stateBounded:
            # normalize to [-1,1]
            scaled_state = (state - self.stateMin)/self.stateRange
            return scaled_state
        return state

    def step(self, action):
        if np.random.uniform(0., 1.) < self.actNoise:
            action = np.random.randint(self.actionDim) if self.actionBound is None \
                else np.random.uniform(-self.actionBound, self.actionBound, self.actionDim)
        state, reward, done, info = self.instance.step(action)
        reward = reward + np.random.normal(0.0, self.reward_noise_sigma)
        if self.stateBounded:
            scaled_state = (state - self.stateMin)/self.stateRange
            return (scaled_state, reward, done, info)
        return (state, reward, done, info)

    def getStateDim(self):
        dim = self.instance.observation_space.shape
        print(dim)
        if len(dim) < 2:
            return dim[0]
        return dim
  
    # this will be the output units in NN
    def getControlDim(self):
        # if discrete action
        if hasattr(self.instance.action_space, 'n'):
            return int(self.instance.action_space.n)
        # if continuous action
        return int(self.instance.action_space.sample().shape[0])

    # Return action ranges
    def getActBound(self):
        #print self.instance.action_space.dtype
        if hasattr(self.instance.action_space, 'high'):
            #self.action_space = spaces.Box(low=self.instance.action_space.low, high=self.instance.action_space.high, shape=self.instance.action_space.low.shape, dtype = np.float64)
            return self.instance.action_space.high[0]
        return None

    # Return action min
    def getActMin(self):
        if hasattr(self.instance.action_space, 'low'):
            return self.instance.action_space.low
        return None

    # Return state range
    ''' the range is rarely used '''
    def getStateRange(self):
        return self.instance.observation_space.high - self.instance.observation_space.low
    
    # Return state min
    def getStateMin(self):
        return self.instance.observation_space.low

    # Close the environment and clear memory
    def close(self):
        self.instance.close()


class HighwayEnvironment(Environment):
    def __init__(self, env_params):
        super(HighwayEnvironment, self).__init__(env_params)
        self.stateDim = self.instance.reset().reshape(-1).shape[0]
        print(' the state dim is :: ========================= ', self.stateDim)

        self.stateRange = self.getStateRange().reshape(-1)
        self.stateMin = self.getStateMin().reshape(-1)

        # action info
        self.actionDim = self.getControlDim()

        # if self.name == 'Acrobot-v1':
        self.statehigh = self.instance.observation_space.high.reshape(-1)
        self.statelow = self.instance.observation_space.low.reshape(-1)

    def step(self, a):
        obs, reward, done, info = self.instance.step(a)
        return obs.reshape(-1), reward, done, info

    def reset(self):
        obs = self.instance.reset()
        return obs.reshape(-1)