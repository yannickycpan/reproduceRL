""" assume a json file (named as actorcritic.json) includes the parameter setting
{
    "environment_names": ["CartPole-v1"],
    "numEvalEpisodes": 1,
    "numTrainEpisodes": 400,
    "evalEverySteps": 1000,
    "maxTotalSamples": 200000,
    "warmUpSteps": 1000,
    "bufferSize": 1000000,
    "batchSize":32,
    "nruns":10,
    "CartPole-v1":{
      "EpisodeSamples": 500,
      "stateBounded": false
    },
    "agent_names": ["LinearActorCritic"],
  "LinearActorCritic":{
    "sweeps": {
      "alpha":[0.1, 0.2, 0.4],
      "critic_factor":[5.0, 10.0],
      "gamma": [1.0],
      "entropy_reg": [0.0, 0.2, 0.8],
      "use_extreme_init": [true],
       "tilings":[8],
      "tiles":[8]
    }
    }
}

call: python main.py actorcritic.json 0
the 0th parameter setting (in this case alpha=0.1, critic_factor = 5, entropy_reg=0.0) will be used.
In this jsonfile, we have "nruns": 10, which means we need to use 10 random seeds for each parameter setting, hence
we should have 3 (alphas) * 2 (critic_factors) * 3 (entropy_regs) * 10 (random seeds) * 3 (algorithms) = 540 independent runs.
As a result, we need to call python example.py actorcritic.json 0, python example.py actorcritic.json 1,
                            ... , python example.py actorcritic.json 539

This can be done by submitting an array of jobs, each job is an independent run.
"""
from utils.rl_exp import Experiment
import numpy as np
import os
import sys
import json
from utils.loggerutils import logger

save_file_format = '%10.6f'


def get_sweep_parameters(parameters, index):
    out = {}
    accum = 1
    for key in sorted(parameters):
        num = len(parameters[key])
        out[key] = parameters[key][int((index / accum) % num)]
        accum *= num
    n_run = int(index / accum)
    n_setting = int(index % accum)
    return out, n_run, n_setting


def get_count_parameters(parameters):
    accum = 1
    for key in parameters:
        num = len(parameters[key])
        accum *= num
    return accum


def merge_agent_params(agent_params, sweep_params):
    for key in sweep_params:
        agent_params[key] = sweep_params[key]
    return agent_params


def supplement_common_params(agent_params, env_params, all_params):
    for key in ["warmUpSteps", "bufferSize", "batchSize", "evalEverySteps",
                "numEvalEpisodes", "maxTotalSamples"]:
        if key not in agent_params:
            if key in env_params:
                agent_params[key] = env_params[key]
            else:
                agent_params[key] = all_params[key]
    agent_params['type'] = env_params['type'] if 'type' in env_params else None
    agent_params['useAtari'] = env_params['useAtari'] if 'useAtari' in env_params else False
    agent_params['envName'] = env_params['name']


# can add if condition in this function if some experiment has special setting
''' the way of checking Atari may need to be modifed in the future '''


def make_experiment(agent_params, env_params, seed, explogger):
    return Experiment(agent_params, env_params, seed, explogger)


def check_if_run_all_atari(params):
    if params['environment_names'][0] == 'AllAtari':
        file = open("atari.names", "r")
        allnames = file.readlines()
        params['environment_names'] = [name.replace('\n', '') for name in allnames]
        for env_name in params['environment_names']:
            params[env_name] = {"EpisodeSamples": 100000, "stateBounded": False}


def save_results(env_name, agent_name, sweep_params, train_lc, eval_lc, eval_ste, n_setting, n_run):
    storedir = env_name + 'results/'
    prefix = storedir + env_name + '_' + agent_name + '_setting_' + str(n_setting) + '_run_' + str(n_run)

    name = prefix + '_EpisodeLC.txt'
    train_lc.tofile(name, sep=',', format=save_file_format)

    name = prefix + '_EpisodeEvalLC.txt'
    eval_lc.tofile(name, sep=',', format=save_file_format)

    name = prefix + '_EpisodeEvalSte.txt'
    eval_ste.tofile(name, sep=',', format=save_file_format)

    params = []
    params_names = '_'
    for key in sorted(sweep_params):
        params.append(sweep_params[key])
        params_names += (key + '_')
    params = np.array(params)
    name = prefix + params_names + 'Params.txt'
    params.tofile(name, sep=',', format=save_file_format)
    return None


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('run as "python run.py [json_file] [index]')
        exit(0)
    # load experiment description from json file
    file = sys.argv[1]
    index = int(sys.argv[2])
    json_dat = open(file, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    ''' check if run all atari games'''
    check_if_run_all_atari(exp)

    accm_total_runs = 0
    last_accm_total_runs = 0
    avg_runs = 30 if 'nruns' not in exp else exp['nruns']
    for env_name in exp['environment_names']:
        for agent_name in exp['agent_names']:
            dirname = env_name + 'results'
            if os.path.exists(dirname) == False:
                os.makedirs(dirname)

            agent_params = exp[agent_name]
            n_total_runs = get_count_parameters(agent_params['sweeps'])*avg_runs
            accm_total_runs += n_total_runs
            if index >= accm_total_runs:
                #print('acc total runs is ', n_total_runs, accm_total_runs)
                last_accm_total_runs = accm_total_runs
                continue
            else:
                index -= last_accm_total_runs

            # environment parameter, i.e. different env need different training time steps
            env_params = exp[env_name]
            env_params['name'] = env_name
            # supplement agent parameter
            supplement_common_params(agent_params, env_params, exp)

            agent_sweep_params, n_run, n_setting = get_sweep_parameters(agent_params['sweeps'], index)
            print('-----------the final corrected index is-------------------', index)
            print('------------setting and run indexes are-----------------', env_name, agent_name, n_setting, n_run)

            agent_params['name'] = agent_name
            agent_params['saveName'] = agent_name if 'saveName' not in agent_params else agent_params['saveName']
            agent_params['sparseReward'] = env_params['sparseReward'] if 'sparseReward' in env_params else False
            agent_params = merge_agent_params(agent_params, agent_sweep_params)
            # create logger
            explogger = logger(agent_params, agent_sweep_params, n_run, n_setting)
            # create experiment
            experiment = make_experiment(agent_params, env_params, n_run, explogger)
            # run experiment and save result
            experiment.run()
            exit(0)
