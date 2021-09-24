#default imports
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

#added imports
from gym_jtr.envs import state_estimator
from gym_jtr.envs import util
import pandas as pd
import time
import math

WINDOW_W = 500
WINDOW_H = 500
ZOOM = 1200

class JtrEnvV0(gym.Env):

  metadata = {'render.modes': ['human']}

  def __init__(self):
    #print('init called')
    super(JtrEnvV0, self).__init__()

    self.start = time.process_time()
    self.viewer = None
    self.state_estimator = state_estimator.StateEstimator(her=False)

    # goal_distance_threshold (float): the threshold after which a goal is considered achieved
    # self.goal_distance_threshold = 10 
    # reset_distance_threshold (float): the threshold of deviation from main route after which the environment is reset
    # self.reset_distance_threshold = 50

    # action space - Rudder
    self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) #scaled between -50, 50

    # observation space - environment
    # all features are scaled to between -1 and 1
    self.observation_space = spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32) #all 29 features + goal location (3)

  def step(self, action):
    observation = self.state_estimator.get_next_observation(action)

    reward, done, diversion = self.state_estimator.get_roman_reward_and_done()
    print('step {}, reward: {}, diversion_roman: {}'.format(self.state_estimator.next_row - (self.state_estimator.start_indice + 61),
                                                reward,
                                                diversion))
    info = {} #not used

    return observation, reward, done, info

  def reset(self):
    #print('reset called')
    self.state_estimator.reset()
    return self.state_estimator.get_current_observation()  # reward, done, info can't be included

  def render(self, mode='human'):
    if self.viewer is None:
        from gym.envs.classic_control import rendering
        self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
        self.viewer.set_bounds(-1, 1, -1, 1)

        real_boat = rendering.make_polygon(v=[[-0.1, -0.1], [-0.1, 0.1], 
                                              [0, 0.2], [0.1, 0.1], [0.1, -0.1]], 
                                          filled=True)
        real_boat.set_color(0.9, 0.9, 1.0)
        self.real_boat_transform = rendering.Transform()
        real_boat.add_attr(self.real_boat_transform)
        self.viewer.add_geom(real_boat)
        
        boat = rendering.make_polygon(v=[[-0.1, -0.1], [-0.1, 0.1], 
                                        [0, 0.2], [0.1, 0.1], [0.1, -0.1]], 
                                      filled=True)
        boat.set_color(0, 0, 1.0)
        self.boat_transform = rendering.Transform()
        boat.add_attr(self.boat_transform)
        self.viewer.add_geom(boat)

    real_boat_loc_array = self.state_estimator.get_real_boat_location()
    real_boat_heading = self.state_estimator.get_real_boat_heading_angle()
    real_boat_lat = util.rescale(real_boat_loc_array[0], 'Latitude')
    real_boat_long = util.to_angle(real_boat_loc_array[1], real_boat_loc_array[2])
    self.real_boat_transform.set_rotation(math.radians(real_boat_heading) + np.pi / 2)
    self.real_boat_transform.set_translation(real_boat_long*ZOOM, real_boat_lat*ZOOM)

    current_state = self.state_estimator.get_current_state()
    heading = util.to_angle(current_state.at['Heading_ov_ground_cos'], current_state.at['Heading_ov_ground_sin'])
    boat_lat = util.rescale(current_state.at['Latitude'], 'Latitude')
    boat_long = util.to_angle(current_state.at['Longitude_cos'], current_state.at['Longitude_sin'])
    self.boat_transform.set_rotation(math.radians(heading) + np.pi / 2)
    self.boat_transform.set_translation(boat_long*ZOOM, boat_lat*ZOOM)

    self.viewer.set_bounds(boat_long*ZOOM-1, boat_long*ZOOM+1, boat_lat*ZOOM-1, boat_lat*ZOOM+1)
    
    print('Rudder: {}'.format(util.rescale(current_state.at['Rudder'], 'Rudder')))
    #print('Heading: {}'.format(heading))
    print('Speed: {}'.format(util.rescale(current_state.at['Speed_ov_ground'], 'Speed_ov_ground')))
    print('Diversion: {}'.format(self.state_estimator.get_diversion_from_real_route()))
    print('Distance to goal: {}'.format(self.state_estimator.get_distance_to_goal()))

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None
    print('Time taken: {}'.format(time.process_time() - self.start))
    
