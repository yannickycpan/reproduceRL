import numpy as np
#import math
import utils.mathutils as mt

'''two ways to use model, one is directly from mujoco engine, another is through gym'''

class MujocoModel(object):
    def __init__(self, path):
        import mujoco_py
        self.model = mujoco_py.load_model_from_path(path)
        self.sim = mujoco_py.MjSim(self.model)
        #print('mujoco model initialized !!!!!!!!!!!!!!!!!!!!!!!! ')

    def do_simulation(self, a):
        self.sim.data.ctrl[:] = a
        for i in range(self.frameskip):
            #print('sim times', i)
            self.sim.step()

    def _get_obs(self):
        return None

    ### does the old act and udd_state matter ??? should one reset sim before set state ???
    def set_state(self, state, posx = None, time = None):
        oldstate = self.sim.get_state()
        # first state is x position, second is none
        posx = 0. if posx is None else posx
        qpos = np.concatenate([np.array([posx]), state[:self.model.nq-1]])
        qvel = state[self.model.nq-1:]
        #time = oldstate.time if time is None else time
        time = 0. if time is None else time
        new_state = mujoco_py.MjSimState(time, qpos, qvel, oldstate.act, oldstate.udd_state)
        #print('udd_state is :: ------------------------- ', oldstate.act)
        self.sim.set_state(new_state)
        self.sim.forward()

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def gymmodel(self, state, a):
        sp, r, gamma = None, None, None
        return sp, r, gamma

'''add two more domains: inverted pendulum and reacher, gym-based implementation'''
class Reacher(MujocoModel):
    def __init__(self, env_params):
        import gym
        self.env = gym.make('Reacher-v2')
        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0
        self.env.reset()

    def termination(self, obs):
        dones = np.zeros(obs.shape[0], dtype = bool)
        if obs.shape[0] == 1:
            return dones[0]

    def gymmodel(self, state, act):
        theta1 = mt.inverse_trig(state[0], state[2])
        theta2 = mt.inverse_trig(state[1], state[3])
        qpos = np.concatenate([np.array([theta1, theta2]), state[4:6]])
        qvel = np.concatenate([state[6:8], np.zeros(2)])
        self.env.env.set_state(qpos, qvel)
        obs, reward, _, _ = self.env.step(act)
        reward = reward + np.random.normal(0.0, self.reward_noise_sigma)
        return obs, reward, None

class InvertedPendulum(MujocoModel):
    def __init__(self, env_params):
        import gym
        self.env = gym.make('InvertedPendulum-v2')
        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0
        self.env.reset()

    def termination(self, obs):
        if len(obs.shape) == 1:
            notdone = np.isfinite(obs).all() and (np.abs(obs[1]) <= .2)
            done = not notdone
        else:
            notdone = np.prod(np.isfinite(obs), axis=1, dtype=bool) * (np.abs(obs[:, 1]) <= .2)
            done = np.logical_not(notdone)
        return done

    def gymmodel(self, state, a):
        qpos = state[:self.env.env.data.qpos.shape[0]]
        qvel = state[self.env.env.data.qpos.shape[0]:]
        self.env.env.set_state(qpos, qvel)
        obs, reward, _, _ = self.env.step(a)
        reward = reward + np.random.normal(0.0, self.reward_noise_sigma)
        gamma = 0.0 if self.termination(obs) else None
        return obs, reward, gamma

class Halfcheetah(MujocoModel):
    def __init__(self, env_params):
        import gym
        self.env = gym.make('HalfCheetah-v2')
        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0
        self.env.reset()

    def termination(self, obs):
        dones = np.zeros(obs.shape[0], dtype=bool)
        if obs.shape[0] == 1:
            return dones[0]

    def gymmodel(self, state, a):
        qpos = np.concatenate([np.array([0.]), state[:8]])
        qvel = state[8:]
        self.env.env.set_state(qpos, qvel)
        obs, reward, _, _ = self.env.step(a)
        reward = reward + np.random.normal(0.0, self.reward_noise_sigma)
        return obs, reward, None

'''its model is inaccurate when all action values are 1'''
class Hopper(MujocoModel):
    def __init__(self, env_params):
        import gym
        self.env = gym.make('Hopper-v2')
        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0
        self.env.reset()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def termination(self, obs):
        if len(obs.shape) == 1:
            return not (np.isfinite(obs).all() and (np.abs(obs[1:]) < 100).all()
                    and (obs[0] > .7) and (abs(obs[1]) < .2))
        else:
            notdone = np.prod(np.isfinite(obs), axis=1, dtype=bool) \
                      * np.all(np.abs(obs[:, 1:]) < 100, axis=1) \
                      * (obs[:, 0] > .7) * (np.abs(obs[:, 1]) < .2)
            done = np.logical_not(notdone)
            return done

    def gymmodel(self, state, a):
        qpos = np.concatenate([np.array([0.]), state[:self.env.env.data.qpos.shape[0]-1]])
        qvel = state[self.env.env.data.qpos.shape[0]-1:]
        self.env.env.set_state(qpos, qvel)
        obs, reward, _, _ = self.env.step(a)
        reward = reward + np.random.normal(0.0, self.reward_noise_sigma)
        gamma = 0. if self.termination(obs) else None
        return obs, reward, gamma

class Swimmer(MujocoModel):
    def __init__(self, env_params):
        import gym
        self.env = gym.make('Swimmer-v2')
        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0
        self.env.reset()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat
        ])

    def termination(self, obs):
        dones = np.zeros(obs.shape[0], dtype=bool)
        if obs.shape[0] == 1:
            return dones[0]

    def gymmodel(self, state, a):
        qpos = np.concatenate([np.array([0.0, 0.0]), state[:self.env.env.data.qpos.shape[0] - 2]])
        qvel = state[self.env.env.data.qpos.shape[0] - 2:]
        self.env.env.set_state(qpos, qvel)
        obs, reward, _, _ = self.env.step(a)
        gamma = 0.0 if self.termination(obs) else None
        reward = reward + np.random.normal(0.0, self.reward_noise_sigma)
        return obs, reward, gamma

class Walker2d(MujocoModel):
    def __init__(self, env_params):
        import gym
        self.env = gym.make('Walker2d-v2')
        self.reward_noise_sigma = env_params['rewardSigma'] if 'rewardSigma' in env_params else 0.0
        self.env.reset()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10., 10.)
        ])

    def termination(self, obs):
        if len(obs.shape) == 1:
            return not (obs[0] > 0.8 and obs[0] < 2.0 and
             obs[1] > -1.0 and obs[1] < 1.0)
        else:
            dones = (obs[:, 0] + 1.25 <= 0.8) + (np.abs(obs[:, 7]) >= 1.0)
            return dones

    def gymmodel(self, state, a, posx = None, time = None):
        qpos = np.concatenate([np.array([0.0]), state[:self.env.env.data.qpos.shape[0] - 1]])
        qvel = state[self.env.env.data.qpos.shape[0] - 1:]
        self.env.env.set_state(qpos, qvel)
        obs, reward, _, _ = self.env.step(a)
        gamma = 0.0 if self.termination(obs) else None
        reward = reward + np.random.normal(0.0, self.reward_noise_sigma)
        return obs, reward, gamma