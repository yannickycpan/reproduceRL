# import utils.mathutils as mt
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from Acrobot import Acrobot
from MountainCar import MountainCar, MountainCarContinuous, MountainCarStop
from CartPole import CartPole
from Pendulumv0 import Pendulumv0
import numpy as np
from MujocoModels import Halfcheetah, Hopper, Swimmer, Walker2d, Reacher, InvertedPendulum
from utils.mathutils import inverse_trig
from modelinfo import ModelInfo

def choose_model(env_params):
    name = env_params['name']
    modelinfo = ModelInfo()
    if name == 'Pendulum-v0':
        modelinfo.termination_condition = termination_invpenduswingup
        modelinfo.model = Pendulumv0().model
    elif name == 'Acrobot-v1':
        modelinfo.termination_condition = termination_acrobot
        modelinfo.model = Acrobot().model
    elif name == 'CartPole-v1':
        modelinfo.termination_condition = termination_cartpole
        modelinfo.model = CartPole().model
    elif name in ['MountainCar-v0', 'MountainCar-Rand']:
        modelinfo.termination_condition = termination_mcar
        modelinfo.model = MountainCar().model
    elif name == 'MountainCarStop':
        modelinfo.termination_condition = termination_mcar
        modelinfo.model = MountainCarStop().model
    elif name == 'MountainCarContinuous-v0':
        modelinfo.termination_condition = termination_mcarconti
        modelinfo.model = MountainCarContinuous().model
    elif name == "RoboschoolAnt-v1":
        modelinfo.termination_condition = termination_ant
        modelinfo.n_binary = 4
    elif name == "RoboschoolWalker2d-v1":
        modelinfo.termination_condition = termination_walker2d
        modelinfo.n_binary = 2
    elif name == "RoboschoolHumanoid-v1":
        modelinfo.termination_condition = termination_humanoid
        modelinfo.n_binary = 2
    elif name == "RoboschoolHopper-v1":
        modelinfo.termination_condition = termination_hopper
        modelinfo.n_binary = 1
    elif name == "RoboschoolHalfCheetah-v1":
        modelinfo.termination_condition = termination_halfcheetah
        modelinfo.n_binary = 6
    elif name == "RoboschoolHalfCheetah-v1":
        modelinfo.termination_condition = termination_halfcheetah
        modelinfo.n_binary = 6
    elif name == "RoboschoolReacher-v1":
        modelinfo.termination_condition = termination_reacher
        modelinfo.n_binary = 0
    elif name == "RoboschoolInvertedPendulum-v1":
        modelinfo.termination_condition = termination_invpendu
        modelinfo.n_binary = 0
    elif name == "RoboschoolInvertedPendulumSwingup-v1":
        modelinfo.termination_condition = termination_invpenduswingup
        modelinfo.n_binary = 0
    elif name == 'HalfCheetah-v2':
        modelinfo.termination_condition = Halfcheetah(env_params).termination
        modelinfo.model = Halfcheetah(env_params).gymmodel
    elif name == 'Hopper-v2':
        modelinfo.termination_condition = Hopper(env_params).termination
        modelinfo.model = Hopper(env_params).gymmodel
    elif name == 'Swimmer-v2':
        modelinfo.termination_condition = Swimmer(env_params).termination
        modelinfo.model = Swimmer(env_params).gymmodel
    elif name == 'Walker2d-v2':
        modelinfo.termination_condition = Walker2d(env_params).termination
        modelinfo.model = Walker2d(env_params).gymmodel
    elif name == 'Reacher-v2':
        modelinfo.termination_condition = Reacher(env_params).termination
        modelinfo.model = Reacher(env_params).gymmodel
    elif name == 'InvertedPendulum-v2':
        modelinfo.termination_condition = InvertedPendulum(env_params).termination
        modelinfo.model = InvertedPendulum(env_params).gymmodel
    #elif name in ["highway-v0", "merge-v0", "roundabout-v0", "parking-v0", "intersection-v0"]:
    #    modelinfo.termination_condition = termination_highway
    return modelinfo

def termination_halfcheetah(s):
    if len(s.shape) == 1:
        return False if s[7] < 1.0 and not s[-1] and not s[-2] and not s[-4] and not s[-5] else True
    else:
        notdones = (s[:, 7] < 1.0) * (s[:, -1] == 0) * (s[:, -2] == 0) * (s[:, -4] == 0) * (s[:, -5] == 0)
        dones = np.logical_not(notdones)
        return dones

def termination_hopper(s):
    if len(s.shape) == 1:
        return True if s[0] + 1.25 <= 0.8 or abs(s[7]) >= 1.0 else False
    else:
        dones = (s[:, 0] + 1.25 <= 0.8) + (np.abs(s[:, 7]) >= 1.0)
        return dones

def termination_walker2d(s):
    if len(s.shape) == 1:
        return True if s[0] + 1.25 <= 0.8 or abs(s[7]) >= 1.0 else False
    else:
        dones = (s[:, 0] + 1.25 <= 0.8) + (np.abs(s[:, 7]) >= 1.0)
        return dones

def termination_ant(s):
    if len(s.shape) == 1:
        return True if s[0] + 0.75 <= 0.26 else False
    else:
        dones = (s[:, 0] + 0.75 <= 0.26)
        return dones

def termination_humanoid(s):
    if len(s.shape) == 1:
        return True if s[0] + 0.8 <= 0.78 else False
    else:
        dones = (s[:, 0] + 0.8 <= 0.78)
        return dones

def termination_gridworld(s):
    if len(s.shape) == 1:
        return True if (s[0] >= 0.95) and (s[1] >= 0.95) else False
    else:
        dones = (s[:, 0] >= 0.95) * (s[:, 1] >= 0.95)
        return dones

def termination_cartpole(s):
    if len(s.shape) == 1:
        return True if np.abs(s[2]) >= 0.41887903/2. \
                   or np.abs(s[0]) >= 4.8/2. else False
    else:
        dones = (np.abs(s[:, 2]) >= 0.41887903/2.) + (np.abs(s[:, 0]) >= 4.8/2.)
        return dones

def termination_acrobot(s):
    if len(s.shape) == 1:
        return True if (-s[0] + s[3]*s[1] - s[2]*s[0]) > 1.0 else False
    else:
        dones = (-s[:, 0] + s[:, 3] * s[:, 1] - s[:, 2] * s[:, 0]) > 1.0
        return dones

def termination_reacher(s):
    return False if len(s.shape) == 1 else np.zeros(s.shape[0], dtype=bool)

def termination_invpendu(s):
    if len(s.shape) == 1:
        theta = inverse_trig(s[2], s[3])
        return np.abs(theta) > .2
    else:
        dones = np.zeros(s.shape[0],dtype=bool)
        for i in range(s.shape[0]):
            dones[i] = True if np.abs(inverse_trig(s[i, 2], s[i, 3])) > .2 else False
        return dones

def termination_invpenduswingup(s):
    return False if len(s.shape) == 1 else np.zeros(s.shape[0],dtype=bool)

def termination_mcar(s):
    if len(s.shape) == 1:
        return True if s[0] > 0.5 else False
    dones = (s[:, 0] > 0.5)
    return dones

def termination_mcarconti(s):
    if len(s.shape) == 1:
        return True if s[0] > 0.45 else False
    dones = (s[:, 0] > 0.45)
    return dones