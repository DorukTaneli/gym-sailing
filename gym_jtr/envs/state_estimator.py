import numpy as np
import pandas as pd
import mlflow
import os
import math
import random

from joblib import Parallel, delayed

from gym_jtr.envs import util
from tensorflow.keras import backend as K


class StateEstimator:

    instance = None

    def __init__(self, her=False,  observation_features=None, parallel=False):
        self.__class__.instance = self
        self.her = her
        self.observation_features = observation_features
        self.parallel = parallel
        self.runtime_minutes = 1 #goal is 1 min ahead

        dirpath = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(dirpath, 'atlantic_usuable_1.csv')
        self.full_data = pd.read_csv(filepath)

        self.boat_states = [
            'AWS', 
            'Yaw_cos', 'Yaw_sin',
            'Pitch', 'AWA_cos', 'AWA_sin',
            'Roll', 'Heading_ov_ground_cos', 'Heading_ov_ground_sin',
            'Speed_ov_ground', 'Heading_Mag_cos', 'Heading_Mag_sin',
            'VMG', 'TWA_cos', 'TWA_sin',
            'Speed_ov_surface', 'Heading_True_cos', 'Heading_True_sin',
            ]
        self.sea_states = [
            'Current_speed', 'Current_direction_cos', 'Current_direction_sin',
            'TWS', 'TWD_cos', 'TWD_sin', 'Air_temp'
            ]

        self._load_models()
        self.reset()

    def reset(self):
        self.start_indice = random.randint(0, 23800)
        self.state_history = self.full_data.iloc[self.start_indice:self.start_indice+61]
        self.next_row = self.start_indice+62
        goal_state = self.full_data.iloc[60*self.runtime_minutes+self.start_indice+61]
        self.desired_goal = np.array([goal_state['Latitude'], goal_state['Longitude_cos'], goal_state['Longitude_sin']])

    def get_current_boat_location(self):
        current_state = self.state_history.tail(1).iloc[0]
        return np.array([current_state.at['Latitude'], 
                         current_state.at['Longitude_cos'], 
                         current_state.at['Longitude_sin']])
    
    def get_real_boat_location(self):
        real_loc = np.array([
            self.full_data.loc[self.next_row-1].at['Latitude'],
            self.full_data.loc[self.next_row-1].at['Longitude_cos'],
            self.full_data.loc[self.next_row-1].at['Longitude_sin'],
            ])
        return real_loc

    def get_real_boat_heading_angle(self):
        heading = util.to_angle(self.full_data.loc[self.next_row-1].at['Heading_ov_ground_cos'],
                                self.full_data.loc[self.next_row-1].at['Heading_ov_ground_sin'])
        return heading

    def _get_current_observation_her(self):
        current_state = self.state_history.tail(1).iloc[0]
        observation = {
            'desired_goal': self.desired_goal,
            'achieved_goal': np.array([current_state.at['Latitude'], current_state.at['Longitude_cos'], current_state.at['Longitude_sin']]),
            'observation': np.squeeze(current_state.to_numpy()),
        }
        return observation

    def get_current_observation_custom(self, feature_list):
        current_state = self.state_history.tail(1).iloc[0]
        obslist = []
        for feature in feature_list:
            obslist.append(current_state.at[feature])
        obslist.append(current_state.at['Latitude'] - self.desired_goal[0])
        obslist.append(current_state.at['Longitude_cos'] - self.desired_goal[1])
        obslist.append(current_state.at['Longitude_sin'] - self.desired_goal[2])
        observation = np.array(obslist, dtype=np.float32)
        return observation

    def get_current_observation(self):
        if self.observation_features is not None:
            return self.get_current_observation_custom(self.observation_features)
        elif self.her:
            return self._get_current_observation_her()
        else:
            current_state = self.state_history.tail(1).iloc[0]
            observation = np.append(np.squeeze(current_state.to_numpy()), 
                                    self.desired_goal)
            return observation

    def get_current_state(self):
        return self.state_history.tail(1).iloc[0]

    def get_next_observation(self, rudder_action):
        #calculate next state
        self.dict = {} #dictionary to hold features

        if self.parallel:
            Parallel(n_jobs=os.cpu_count(), require='sharedmem')(
                delayed(predict_next_states)(target) for target in self.boat_states)
        else:
            for target in self.boat_states: #predict next boat states
                time_series = util.time_series_gen(data=self.state_history, target=target, batch_size=1, fs=1)
                self.dict[target] = np.squeeze(self.models_dict["{}model".format(target)].predict(time_series))
        K.clear_session()

        for sea_state in self.sea_states: #add next sea states
            self.dict[sea_state] = self.full_data.at[self.next_row-1, sea_state]
        
        self.dict["Rudder"] = rudder_action #add rudder action

        # calculate next lat-long
        current_state = self.state_history.tail(1).iloc[0]
        self.dict["Latitude"], self.dict["Longitude_cos"], self.dict["Longitude_sin"] = util.calculate_next_location(
           current_state.at['Latitude'], current_state.at['Longitude_cos'], current_state.at['Longitude_sin'], 
           current_state.at['Speed_ov_ground'], current_state.at['Heading_ov_ground_cos'], current_state.at['Heading_ov_ground_sin'],
        )

        #append next state
        next_state = pd.DataFrame(data=self.dict, index=[self.next_row])
        self.state_history = self.state_history.append(next_state)

        #drop oldest row
        self.state_history.drop(self.state_history.head(1).index, inplace=True)

        self.next_row += 1

        return self.get_current_observation()

    def _load_models(self):
        dirpath = os.path.dirname(os.path.abspath(__file__))
        self.models_dict = {}
        for target in self.boat_states:
            self.models_dict["{}model".format(target)] = mlflow.pyfunc.load_model(
                os.path.join(dirpath, 'models/{}/artifacts/{}model'.format(target, target)))

    def get_last_5_steps(self):
        return self.state_history.tail(5)

    def get_roman_reward_and_done(self):
        steps_taken = self.next_row - (self.start_indice + 61)
        current_state = self.state_history.tail(1).iloc[0]
        R = 6378*1000
        #real boat loc
        way_p = (util.rescale(self.full_data.loc[self.next_row-1].at['Latitude'], 'Latitude'),
                 util.to_angle(self.full_data.loc[self.next_row-1].at['Longitude_cos'],
                               self.full_data.loc[self.next_row-1].at['Longitude_sin']))
        lat_2 = util.rescale(current_state.at['Latitude'], 'Latitude')
        long_2 = util.to_angle(current_state.at['Longitude_cos'], current_state.at['Longitude_sin'])
        del_long = abs(way_p[1] - long_2)
        # cosine of the angle between the 2 vectors:
        # 2. From centre of the earth to current positioin
        # 1. From centre of the Earth to the current way point
        cent_cos = math.sin(math.radians(lat_2)) *\
            math.sin(math.radians(way_p[0])) +\
            math.cos(math.radians(lat_2)) *\
            math.cos(math.radians(way_p[0])) *\
            math.cos(math.radians(del_long))
        
        #cent_cos = np.round(cent_cos, decimals=4)
        ang_cent = math.acos(cent_cos)
        diversion = R*ang_cent  # diversion of the boat from original course

        # reciprocal reward
        gamma = 10
        if diversion == 0:
            reward = 100
        else:
            reward = gamma * 1/(diversion/1000)

        done = False
        if diversion > 100 or steps_taken > 3600:
            done = True
        
        return reward, done, diversion


    def get_diversion_from_real_route(self):
        return util.calculate_distance(self.get_real_boat_location(), 
                                       self.get_current_boat_location())

    def get_distance_to_goal(self):
        return util.calculate_distance(self.desired_goal, 
                                       self.get_current_boat_location())

'''
        if self.parallel:
            pool = multiprocessing.Pool()
            pool.map(partial(predict_next_states, dict=dict,
                             state_history=self.state_history,
                             models_dict=self.models_dict), 
                     self.boat_states)
'''

def predict_next_states(target):
    se = StateEstimator.instance
    time_series = util.time_series_gen(data=se.state_history, target=target, batch_size=1, fs=1)
    se.dict[target] = np.squeeze(se.models_dict["{}model".format(target)].predict(time_series))
