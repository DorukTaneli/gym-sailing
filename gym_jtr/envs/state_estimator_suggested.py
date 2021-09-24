'''
Suggested State Estimator Interface to be implemented and used
when boat state estimator models are improved.

You can check state_estimator.py and jtr_env_modelless_v4 for examples,
but the implementation will differ depending on 
how the new boat state estimator works
'''

import numpy as np
import random
from gym_jtr.envs import util

class StateEstimatorSuggested:

    def __init__(self):
        #how far the goal is for each episode
        self.runtime_minutes = random.randint(1, 5)
        #store desired goal
        self.desired_goal = None
        
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_current_boat_location(self) -> np.array:
        raise NotImplementedError
    
    def get_real_boat_location(self) -> np.array:   
        raise NotImplementedError

    def get_real_boat_heading_angle(self):
        raise NotImplementedError

    def get_current_observation(self):
        raise NotImplementedError

    def get_current_state(self):
        raise NotImplementedError

    def get_next_observation(self, rudder_action):
        raise NotImplementedError

    def get_diversion_from_real_route(self):
        return util.calculate_distance(self.get_real_boat_location(), 
                                       self.get_current_boat_location())

    def get_distance_to_goal(self):
        return util.calculate_distance(self.desired_goal,
                                       self.get_current_boat_location())
