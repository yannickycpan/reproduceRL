import numpy as np
import random
from environments.environments import Environment
from environments.GridWorld import GridWorld, GridWorldContinuous, MazeGridWorld, \
    TwoGoalGridWorld
from environments.MountainCar import MountainCar, MountainCarStop
import time

def make_environment(env_params):
    env_name = env_params['name']
    env_params['qnntype'] = 'Regular'
    env_params['useAtari'] = False
    env_params['statescale'] = 1.0
    if env_name in ['MountainCar-v0', 'MountainCar-Rand']:
        return MountainCar(env_params)
    elif env_name in ["highway-v0", "merge-v0", "roundabout-v0", "parking-v0", "intersection-v0"]:
        from environments.environments import HighwayEnvironment
        return HighwayEnvironment(env_params)
    else:
        return Environment(env_params)


def make_environment(env_params):
    env_name = env_params['name']
    env_params['qnntype'] = 'Regular'
    env_params['useAtari'] = False
    env_params['statescale'] = 1.0
    if env_name in ['ToughGridWorld', 'GridWorld', 'GridWorld-Rand']:
        return GridWorld(env_params)
    elif env_name == 'MazeGridWorld' or env_name == 'ContinuousMazeGridWorld':
        return MazeGridWorld(env_params)
    elif env_name == 'TwoGoalGridWorld':
        return TwoGoalGridWorld(env_params)
    elif env_name == 'GridWorldContinuous':
        return GridWorldContinuous(env_params)
    elif env_name in ['MountainCar-v0', 'MountainCar-Rand']:
        return MountainCar(env_params)
    elif env_name == 'MountainCarStop':
        return MountainCarStop(env_params)
    elif 'NoFrameskip-v4' in env_name:
        print('no frameskip is used ----------------------- ')
        from environments.environments import AtariEnvironment
        env_params['type'] = 'Atari'
        env_params['qnntype'] = 'Atari'
        env_params['useAtari'] = True
        env_params['statescale'] = 255.0
        return AtariEnvironment(env_params)
    elif env_name in ['space_invaders', 'seaquest', 'freeway', 'breakout', 'asterix']:
        print('MINATARI is used -----------------------============================ ')
        from environments.environments import MinAtariEnvironment
        env_params['qnntype'] = 'MinAtari'
        env_params['useAtari'] = True
        return MinAtariEnvironment(env_params)
    elif env_name in ["highway-v0", "merge-v0", "roundabout-v0", "parking-v0", "intersection-v0"]:
        from environments.environments import HighwayEnvironment
        return HighwayEnvironment(env_params)
    elif 'Disc' in env_name:
        from environments.environments import DiscretizedContinuous
        return DiscretizedContinuous(env_params)
    else:
        return Environment(env_params)


def make_agent(agent_params):
    agent_name = agent_params['name']
    if agent_name in ["DQN", "NoTargetDQN"]:
        from agents.DQN import DQNAgent
        return DQNAgent(agent_params)
    elif "DQNSpReg" in agent_name:
        from agents.DQNSpReg import DQNSpRegAgent
        return DQNSpRegAgent(agent_params)
    elif "TCDQN" in agent_name:
        from agents.TCDQN import TCDQNAgent
        return TCDQNAgent(agent_params)
    elif "TCNN" in agent_name:
        from agents.TCNN import TCNNAgent
        return TCNNAgent(agent_params)
    elif agent_name in ["DDPG", "NoTargetDDPG"]:
        from agents.DDPG import DDPGAgent
        return DDPGAgent(agent_params)
    elif "TCDDPG" in agent_name:
        from agents.TCDDPG import TCDDPGAgent
        return TCDDPGAgent(agent_params)
    elif 'ModelDQN' in agent_name:
        from agents.ModelDQN import ModelDQNAgent
        return ModelDQNAgent(agent_params)
    elif 'ModelDDPG' in agent_name:
        from agents.ModelDDPG import ModelDDPGAgent
        return ModelDDPGAgent(agent_params)
    else:
        print("agent not found!!!")
        exit(0)


def supplement_agent_params(agent_params, env, envparams_dict):
    agent_params['stateDim'] = env.stateDim
    agent_params['actionDim'] = env.actionDim
    agent_params['actionBound'] = env.actionBound
    agent_params['stateHigh'] = env.statehigh
    agent_params['stateLow'] = env.statelow
    agent_params['statescale'] = envparams_dict['statescale']
    agent_params['qnntype'] = envparams_dict['qnntype']
    agent_params['useAtari'] = envparams_dict['useAtari']
    return agent_params


def supplement_env_params(agent_params, env_params):
    for param in ['rewardSigma', 'actNoise', 'EnvStepSize']:
        if param in agent_params:
            env_params[param] = agent_params[param]
    return env_params


class Experiment(object):
    def __init__(self, agent_params, env_params, seed, explogger):
        agent_params['seed'] = seed
        env_params['seed'] = seed
        np.random.seed(seed)
        random.seed(seed)

        self.logger = explogger

        agent_params['envName'] = env_params['name']

        self.env_name = env_params['name']
        self.statebounded = agent_params['stateBounded'] = env_params['stateBounded']

        env_params = supplement_env_params(agent_params, env_params)
        self.environment = make_environment(env_params)

        agent_params = supplement_agent_params(agent_params, self.environment, env_params)
        agent_params['modelInfo'] = self.environment.modelinfo

        self.model = self.environment.modelinfo.model
        self.terminal_condition = self.environment.modelinfo.termination_condition
        self.agent = make_agent(agent_params)

        self.MaxEpisodeSteps = self.environment.EPISODE_STEPS_LIMIT
        #self.NumEpisodes = agent_params['numTrainEpisodes']
        self.NumEvalEpisodes = agent_params['numEvalEpisodes']
        self.EvalEverySteps = agent_params['evalEverySteps']
        self.NumTotalSamples = agent_params['maxTotalSamples']

        self.sparseReward = env_params['sparseReward'] if 'sparseReward' in env_params else False

        ''' a seperate env for evaluation '''
        self.eval_environment = make_environment(env_params)

        self.train_step_rewards = []
        self.eval_step_rewards = []
        self.train_episode_rewards = []
        self.eval_episode_rewards = []
        self.eval_episode_rewards_ste = []
        self.accum_reward = 0.
        self.episode = 0
        self.sampleCount = 0

        self.starttime = time.time()
        self.maxtime = agent_params['maxTime'] if 'maxTime' in agent_params else None

        print(' the number of total samples is ==================================== ', self.NumTotalSamples, self.maxtime)

    def check_outof_time(self):
        if self.maxtime is None:
            return False
        if time.time() - self.starttime >= self.maxtime:
            return True

    def TrainEpisode(self):
        episode_reward = 0.
        step = 0
        done = False
        obs = self.environment.reset()
        #print(obs)
        act = self.agent.take_action(obs)
        newEpisode = True
        #print('new episode start ------------------------------------------------------------------------------------------------ ')
        while not (done or step == self.MaxEpisodeSteps or self.sampleCount == self.NumTotalSamples or self.check_outof_time()):
            if self.sampleCount % self.EvalEverySteps == 0:
                self.EvalRun()
            obs_n, reward, done, info = self.environment.step(act)
            done = self.terminal_condition(obs_n) if self.terminal_condition is not None else done
            info = {} if info is None else info
            info['EpisodeEnd'] = True if (step+1 == self.MaxEpisodeSteps) or done else False
            self.accum_reward += reward
            episode_reward += reward
            self.agent.update(obs, act, obs_n, float(reward), done, info)
            act = self.agent.take_action(obs_n)
            obs = obs_n
            step += 1
            self.sampleCount += 1
        return episode_reward, step

    def EvalEpisode(self):
        episode_reward = 0.
        step = 0
        optactcount = 0
        done = False
        obs = self.eval_environment.reset()
        act = self.agent.take_action_eval(obs)
        optactcount += (act==1)
        avgspeed = []
        while not (done or step == self.MaxEpisodeSteps):
            obs_n, reward, done, info = self.eval_environment.step(act)
            done = self.terminal_condition(obs_n) if self.terminal_condition is not None else done
            episode_reward += reward
            act = self.agent.take_action_eval(obs_n)
            step += 1
            optactcount += (act == 1)
            if self.env_name in ["intersection-v0", "merge-v0", "roundabout-v0", "highway-v0", "parking-v0"]:
                if 'crashed' in info:
                    if 'NumCrashEvalLC' not in self.logger.logger_dict:
                        self.logger.logger_dict['NumCrashEvalLC'] = []
                    self.logger.logger_dict['NumCrashEvalLC'].append(int(info['crashed']))
                if 'speed' in info:
                    avgspeed.append(info['speed'])
        episode_reward = optactcount/(step + 1.) if self.env_name == 'RiverSwim' else episode_reward
        print('eval episode reward is --------------------------- ', episode_reward)
        if self.env_name in ["intersection-v0", "merge-v0", "roundabout-v0", "highway-v0", "parking-v0"]:
            if 'SpeedEvalLC' not in self.logger.logger_dict:
                self.logger.logger_dict['SpeedEvalLC'] = []
            self.logger.logger_dict['SpeedEvalLC'].append(np.mean(avgspeed))
            print('the avg speed is --------------------------- ', np.mean(avgspeed))
        return episode_reward, -step

    def EvalRun(self):
        '''start evaluation episodes'''
        eval_rewards = np.zeros(self.NumEvalEpisodes)
        negsteps = np.zeros(self.NumEvalEpisodes)
        for epi in range(self.NumEvalEpisodes):
            eval_rewards[epi], negsteps[epi] = self.EvalEpisode()
        self.logger.logger_dict['EpisodeEvalLC'].append(np.max(eval_rewards))
        if self.sparseReward or self.eval_environment.reward_noise_sigma > 0.:
            self.logger.logger_dict['EpisodeEvalNegStepsLC'].append(np.max(negsteps))
        self.agent.custermized_log(self.logger)
        self.logger.save_results()

    def run(self):
        every = 1
        episode = 0
        while self.sampleCount < self.NumTotalSamples and not self.check_outof_time():
            reward, num_steps = self.TrainEpisode()
            self.logger.logger_dict['EpisodeLC'].append(reward)
            if self.sparseReward:
                self.logger.logger_dict['EpisodeNegStepsLC'].append(-num_steps)
            if episode % every == 0:
                print("ep: " + str(episode) + ", r: " + str(reward), "num steps: ", str(num_steps))
                if len(self.logger.logger_dict['EpisodeEvalLC']) > 1:
                    print("eval r and sample n: " + str(self.logger.logger_dict['EpisodeEvalLC'][-1]), self.sampleCount)
            episode += 1
        self.environment.close()
        if self.eval_environment is not None:
            self.eval_environment.close()
