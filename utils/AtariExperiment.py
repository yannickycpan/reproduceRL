from utils.rl_exp import Experiment
import numpy as np
#import psutil
#import os
#import sys
import time


def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes)+"B"
    elif abs(bytes) < 1e6:
        return str(round(bytes/1e3, 2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"

'''
def get_process_memory():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss, mi.vms
'''


def record_m(start, end):
    return (end - start)/60.


class Atari(Experiment):
    def __init__(self, agent_params, env_params, seed, explogger):
        super(Atari, self).__init__(agent_params, env_params, seed, explogger)
        self.eval_sample_count = 0
        self.total_eval_samples = 150000
        self.max_eval_minutes = 5
        self.maxEvalEpisodes = 3
        #self.total_eval_samples = int(1350/4)
        #self.EvalEverySteps = int(250000/4)

    def log_mem(self):
        # print(' physical memory usage is :: -------------------------------------- ', format_bytes(rss_after - rss_before))
        # print( ' virtual memory usage is :: -------------------------------------- ', format_bytes(vms_after - vms_before))
        # print(' number of training steps is :: ----------------------------------- ', num_steps, self.sampleCount)
        return

    def run(self):
        every = 1
        episode = 0
        while self.sampleCount < self.NumTotalSamples:
            # runs a single episode and returns the accumulated reward for that episode
            #rss_before, vms_before = get_process_memory()
            reward, num_steps = self.TrainEpisode()
            #rss_after, vms_after = get_process_memory()
            self.train_episode_rewards.append(reward)
            self.logger.logger_dict['EpisodeLC'].append(reward)

            if episode % every == 0:
                print("ep: " + str(episode) + ", r: " + str(reward), "num steps: ", str(num_steps))
                if len(self.logger.logger_dict['EpisodeEvalLC']) > 1:
                    print("eval r and sample n: " + str(self.logger.logger_dict['EpisodeEvalLC'][-1]), self.sampleCount)
            episode += 1
        self.environment.close()
        if self.eval_environment is not None:
            self.eval_environment.close()

    # Runs a single episode
    def TrainEpisode(self):
        episode_reward = 0.
        step = 0
        skipcount = 0
        done = False
        obs = self.environment.reset()
        act = self.agent.take_action(np.array(obs)[None, :])
        while not (done or step == self.MaxEpisodeSteps or self.sampleCount == self.NumTotalSamples):
            if self.sampleCount % self.EvalEverySteps == 0:
                self.EvalRun()
                # print('size of memory of a state is :: ', sys.getsizeof(np.array(obs)))
            obs_n, reward, done, info = self.environment.step(act)
            info = {} if info is None else info
            info['EpisodeEnd'] = True if (step + 1 == self.MaxEpisodeSteps) or done else False
            self.accum_reward += reward
            episode_reward += reward

            self.agent.update(obs, act, obs_n, float(reward), done, info)
            skipcount += 1
            act = self.agent.take_action(np.array(obs_n)[None, :]) if skipcount % 4 == 0 else act
            obs = obs_n
            step += 1
            self.sampleCount += 1
        return episode_reward, step


    def EvalRun(self):
        '''start evaluation episodes'''
        eval_rewards = []
        self.eval_sample_count = 0
        start_time = time.time()
        end_time = start_time
        while self.eval_sample_count < self.total_eval_samples and \
                record_m(start_time, end_time) <= self.max_eval_minutes:
            if len(eval_rewards) >= self.maxEvalEpisodes:
                break
            eval_reward = self.EvalEpisode()
            eval_rewards.append(eval_reward)
            end_time = time.time()

        self.logger.logger_dict['EpisodeEvalLC'].append(np.max(eval_rewards))
        # this ste will not be used
        self.logger.logger_dict['EpisodeEvalSte'].append(np.std(eval_rewards) / np.sqrt(len(eval_rewards)))
        self.agent.custermized_log(self.logger)
        self.logger.save_results()


    def EvalEpisode(self):
        episode_reward = 0.
        step = 0
        done = False
        obs = self.eval_environment.reset()
        act = self.agent.take_action_eval(np.array(obs)[None, :])
        start_time = time.time()
        end_time = start_time
        while not (done or step == self.MaxEpisodeSteps or self.eval_sample_count >= self.total_eval_samples or
                   record_m(start_time, end_time) >= self.max_eval_minutes):
            obs_n, reward, done, info = self.eval_environment.step(act)
            episode_reward += reward
            act = self.agent.take_action_eval(np.array(obs_n)[None, :])
            step += 1
            self.eval_sample_count += 1
            end_time = time.time()
            #print('time cost in minunte is :: ', record_m(start_time, end_time))
        #print(' finish evaluating policy :: ', episode_reward)
        return episode_reward