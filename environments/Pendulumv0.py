import numpy as np
from utils.mathutils import inverse_trig

class Pendulumv0(object):
    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None

        self.state = None

        high = np.array([1., 1., self.max_speed])
        self.statehigh = high
        self.statelow = -high

    def _get_s(self, obs):
        s = np.zeros(2)
        s[1] = obs[2]
        s[0] = inverse_trig(obs[0], obs[1])
        return s

    def model(self, obs, u):
        if obs is not None:
            #then it is an obs, and convert it to state
            s = self._get_s(obs)
        else:
            s = self.state.copy()

        th, thdot = s[0], s[1] # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        reward = -costs
        if obs is not None:
            return self._get_obs(np.array([newth, newthdot])), reward, None
        return np.array([newth, newthdot]), reward, None

    def step(self, a):
        sp, reward, _ = self.model(obs=None, u = a)
        self.state = sp.copy()
        return self._get_obs(self.state), reward, False, None

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = np.random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs(self.state)

    def _get_obs(self, state):
        theta, thetadot = state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)