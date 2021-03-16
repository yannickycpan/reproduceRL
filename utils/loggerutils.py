import os
import numpy as np
import time

save_file_format = '%10.6f'

class logger(object):
    def __init__(self, agent_params, agent_sweep_params, n_run, n_setting):
        self.env_name = agent_params['envName']
        self.agent_name = agent_params['saveName']
        self.use_atari_nn = agent_params['useAtari']
        self.sparse_reward = agent_params['sparseReward']

        self.n_setting = n_setting
        self.n_run = n_run
        self.sweep_params = agent_sweep_params
        self.agent_params = agent_params
        self.logger_dict = {'EpisodeLC':[], 'EpisodeEvalLC':[], 'EpisodeEvalSte':[],
                            'EpisodeEvalNegStepsLC':[], 'EpisodeNegStepsLC':[], 'timecost': []}
        self.dirname = self.env_name + 'results'
        if os.path.exists(self.dirname) == False:
            os.makedirs(self.dirname)
        self.dirname += '/'
        self.paramsave_indicator = False

        self.starttime = time.time()
        self.endtime = 0.0


    def save_all_params_to_json(self, name, dict):
        import json
        newdict = {}
        def is_jsonable(x):
            try:
                json.dumps(x)
                return True
            except:
                return False
        for key in dict:
            if is_jsonable(dict[key]):
                newdict[key] = dict[key]
        with open(name, 'w') as fp:
            json.dump(newdict, fp)

    ''' this is no longer needed '''
    def save_sweep_params_only(self, sweep_params, prefix):
        params = []
        params_names = '_'
        for key in sorted(sweep_params):
            params.append(sweep_params[key])
            params_names += (key + '_')
        params = np.array(params)
        name = prefix + params_names + 'Params.txt'
        params.tofile(name, sep=',', format=save_file_format)
        return None


    def calculate_accreward(self, prefix):
        if len(self.logger_dict['EpisodeEvalLC']) > 0:
            if 'AccumuEpisodeEvalLC' not in self.logger_dict:
                self.logger_dict['AccumuEpisodeEvalLC'] = []
                self.logger_dict['AccumuEpisodeEvalLC'].append(self.logger_dict['EpisodeEvalLC'][-1])
            else:
                temp = self.logger_dict['AccumuEpisodeEvalLC'][-1] + self.logger_dict['EpisodeEvalLC'][-1]
                self.logger_dict['AccumuEpisodeEvalLC'].append(temp)
            name = prefix + '_AccumuEpisodeEvalLC' + '.txt'
            np.array(self.logger_dict['AccumuEpisodeEvalLC']).tofile(name, sep=',', format=save_file_format)


    def save_results(self):
        prefix = self.dirname + self.env_name + '_' + self.agent_name \
                 + '_setting_' + str(self.n_setting) + '_run_' + str(self.n_run)

        if not self.use_atari_nn and self.sparse_reward:
            self.calculate_accreward(prefix)

        self.endtime = time.time()
        self.logger_dict['timecost'].append(self.endtime - self.starttime)
        print('the current time cost is ========================= ', self.logger_dict['timecost'][-1])

        for key in self.logger_dict:
            if len(self.logger_dict[key]) > 0:
                name = prefix + '_' + key + '.txt'
                np.array(self.logger_dict[key]).tofile(name, sep=',', format=save_file_format)

        if not self.paramsave_indicator:
            param_filename = self.dirname + self.env_name + '_' + self.agent_name \
                           + '_setting_' + str(self.n_setting)  + '_Params.json'
            self.save_all_params_to_json(param_filename, self.agent_params)
            # self.save_sweep_params_only(self.sweep_params, prefix)
            self.paramsave_indicator = True
