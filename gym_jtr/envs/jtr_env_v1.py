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

class JtrEnvV1(gym.Env):

  metadata = {'render.modes': ['human', 'console']}

  def __init__(self):
    super(JtrEnvV1, self).__init__()
    self.viewer = None

    # goal_distance_threshold (float): the threshold after which a goal is considered achieved
    self.goal_distance_threshold = 10 
    # reset_distance_threshold (float): the threshold of deviation from main route after which the environment is reset
    self.reset_distance_threshold = 25

    # action space - Rudder
    self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) #scaled between -50, 50

    # observation space
    observation_features = [
      'Speed_ov_ground',
      'Heading_ov_ground_cos',
      'Heading_ov_ground_sin',
    ]

    # all features are scaled to between -1 and 1
    # observation space is the observation_features + 3 features of location relative to goal
    self.observation_space = spaces.Box(low=-1, high=1, shape=(len(observation_features)+3,), dtype=np.float32)

    self.state_estimator = state_estimator.StateEstimator(observation_features=observation_features)
    self.start = time.process_time()

  def step(self, action):
    observation = self.state_estimator.get_next_observation(action)

    done = False
    if (self.state_estimator.get_diversion_from_real_route() > self.reset_distance_threshold or
        self.state_estimator.get_distance_to_goal() < self.goal_distance_threshold):
      done = True

    if self.state_estimator.get_distance_to_goal() < self.goal_distance_threshold:
      reward = self.reward_per_meter * self.closest_to_goal
    else:
      current_goal_distance = self.state_estimator.get_distance_to_goal()
      if current_goal_distance < self.closest_to_goal:
        reward = self.reward_per_meter * (self.closest_to_goal - current_goal_distance)
        self.closest_to_goal = current_goal_distance
      else:
        reward = 0

    reward -= self.penalty_per_step

    info = {} #not used

    '''
    step = self.state_estimator.next_row - (self.state_estimator.start_indice + 61)
    diversion = self.state_estimator.get_diversion_from_real_route()
    goal_distance = self.state_estimator.get_distance_to_goal()
    print('step {}, reward: {}'.format(step, reward))
    print('diversion: {}, distance to goal: {}'.format(diversion, goal_distance))
    '''
    return observation, reward, done, info


  def reset(self):
    self.state_estimator.reset()
    real_steps = self.state_estimator.runtime_minutes*60
    self.penalty_per_step = 100/real_steps
    self.initial_goal_distance = self.state_estimator.get_distance_to_goal()
    self.reward_per_meter = 1000/self.initial_goal_distance
    self.closest_to_goal = self.initial_goal_distance

    if self.viewer:
      self.viewer.close()
      self.viewer = None

    return self.state_estimator.get_current_observation()  # reward, done, info can't be included


  def render(self, mode='human'):
    if mode == 'console':
        current_state = self.state_estimator.get_current_state()
        print('Rudder: {}'.format(util.rescale(current_state.at['Rudder'], 'Rudder')))
        print('Speed: {}'.format(util.rescale(current_state.at['Speed_ov_ground'], 'Speed_ov_ground')))
        print('Diversion: {}'.format(self.state_estimator.get_diversion_from_real_route()))
        print('Distance to goal: {}'.format(self.state_estimator.get_distance_to_goal()))
        return

    from gym.envs.classic_control import rendering
    if self.viewer is None:
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

        current_state = self.state_estimator.get_current_state()
        last_boat_lat = util.rescale(current_state.at['Latitude'], 'Latitude')
        last_boat_long = util.to_angle(current_state.at['Longitude_cos'], current_state.at['Longitude_sin'])
        self.last_boat_loc = (last_boat_long*ZOOM, last_boat_lat*ZOOM)
        real_boat_loc_array = self.state_estimator.get_real_boat_location()
        last_real_boat_lat = util.rescale(real_boat_loc_array[0], 'Latitude')
        last_real_boat_long = util.to_angle(real_boat_loc_array[1], real_boat_loc_array[2])
        self.last_real_boat_loc= (last_real_boat_long*ZOOM, last_real_boat_lat*ZOOM)

    real_boat_loc_array = self.state_estimator.get_real_boat_location()
    real_boat_heading = self.state_estimator.get_real_boat_heading_angle()
    real_boat_lat = util.rescale(real_boat_loc_array[0], 'Latitude')
    real_boat_long = util.to_angle(real_boat_loc_array[1], real_boat_loc_array[2])
    self.real_boat_transform.set_rotation(-math.radians(real_boat_heading))
    self.real_boat_transform.set_translation(real_boat_long*ZOOM, real_boat_lat*ZOOM)

    real_boat_loc = (real_boat_long*ZOOM, real_boat_lat*ZOOM)
    real_path_part = rendering.Line(real_boat_loc, self.last_real_boat_loc)
    real_path_part.set_color(0.9, 0.9, 1.0)
    real_path_transform = rendering.Transform()
    real_path_part.add_attr(real_path_transform)
    self.viewer.add_geom(real_path_part)
    self.last_real_boat_loc = real_boat_loc

    current_state = self.state_estimator.get_current_state()
    heading = util.to_angle(current_state.at['Heading_ov_ground_cos'], current_state.at['Heading_ov_ground_sin'])
    boat_lat = util.rescale(current_state.at['Latitude'], 'Latitude')
    boat_long = util.to_angle(current_state.at['Longitude_cos'], current_state.at['Longitude_sin'])
    self.boat_transform.set_rotation(-math.radians(heading))
    self.boat_transform.set_translation(boat_long*ZOOM, boat_lat*ZOOM)

    boat_loc = (boat_long*ZOOM, boat_lat*ZOOM)
    path_part = rendering.Line(boat_loc, self.last_boat_loc)
    path_part.set_color(0, 0, 1.0)
    self.viewer.add_geom(path_part)
    self.last_boat_loc = boat_loc

    self.viewer.set_bounds(boat_long*ZOOM-1, boat_long*ZOOM+1, boat_lat*ZOOM-1, boat_lat*ZOOM+1)
    
    print('Rudder: {}'.format(util.rescale(current_state.at['Rudder'], 'Rudder')))
    print('Speed: {}'.format(util.rescale(current_state.at['Speed_ov_ground'], 'Speed_ov_ground')))
    print('Diversion: {}'.format(self.state_estimator.get_diversion_from_real_route()))
    print('Distance to goal: {}'.format(self.state_estimator.get_distance_to_goal()))

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')


  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None
    print('Time taken: {}'.format(time.process_time() - self.start))
    
