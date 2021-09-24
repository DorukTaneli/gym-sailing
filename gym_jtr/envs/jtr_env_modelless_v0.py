#default imports
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

#added imports
from gym_jtr.envs import util
import pandas as pd
import os
import random
import time
import math

WINDOW_W = 500
WINDOW_H = 500
ZOOM = 1200

class JtrEnvModellessV0(gym.Env):

  metadata = {'render.modes': ['human', 'console']}

  def __init__(self):
    super(JtrEnvModellessV0, self).__init__()
    self.viewer = None
    self.runtime_minutes = 1

    # goal_distance_threshold (float): the threshold after which a goal is considered achieved
    self.goal_distance_threshold = 10 
    # reset_distance_threshold (float): the threshold of deviation from main route after which the environment is reset
    self.reset_distance_threshold = 50
    # the timesteps to wait after real boat reaches location before the environment is reset
    self.reset_timestep_threshold = 15

    # action space - Rudder
    self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) #scaled between -50, 50

    # observation space
    self.observation_features = [
      'Speed_ov_ground',
      'Heading_ov_ground_cos',
      'Heading_ov_ground_sin',
    ]

    # all features are scaled to between -1 and 1
    # observation space is the observation_features + 3 features of location relative to goal
    self.observation_space = spaces.Box(low=-1, high=1, shape=(len(self.observation_features)+3,), dtype=np.float32)

    self.start = time.process_time()

    #added for jtr_env_test
    dirpath = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dirpath, 'atlantic_usuable_1.csv')
    self.full_data = pd.read_csv(filepath)

  def step(self, action):
    observation = self.get_next_observation(action)
    step = self.next_row - (self.start_indice + 61)

    done = False
    if (self.get_diversion_from_real_route() > self.reset_distance_threshold or
        self.get_distance_to_goal() < self.goal_distance_threshold or
        step > self.real_steps + self.reset_timestep_threshold):
      done = True

    if self.get_distance_to_goal() < self.goal_distance_threshold:
      reward = self.reward_per_meter * self.closest_to_goal
    else:
      current_goal_distance = self.get_distance_to_goal()
      if current_goal_distance < self.closest_to_goal:
        reward = self.reward_per_meter * (self.closest_to_goal - current_goal_distance)
        self.closest_to_goal = current_goal_distance
      else:
        reward = 0

    reward -= self.penalty_per_step
    if (self.get_diversion_from_real_route() > self.reset_distance_threshold or
        step > self.real_steps + self.reset_timestep_threshold):
      reward -= 300

    self.total_reward += reward

    info = {} #not used

    '''
    if done:
      diversion = self.get_diversion_from_real_route()
      goal_distance = self.get_distance_to_goal()
      print('step {}, total_reward: {}'.format(step, self.total_reward))
      print('diversion: {}, distance to goal: {}'.format(diversion, goal_distance))
      print("last observation: {}".format(observation))
    '''
    
    return observation, reward, done, info


  def reset(self):
    self.start_indice = random.randint(0, 23800)
    self.state_history = self.full_data.iloc[self.start_indice:self.start_indice+61]
    self.full_history = self.state_history
    self.next_row = self.start_indice+62
    goal_state = self.full_data.iloc[60*self.runtime_minutes+self.start_indice+61]
    self.desired_goal = np.array([goal_state['Latitude'], goal_state['Longitude_cos'], goal_state['Longitude_sin']])

    self.real_steps = self.runtime_minutes*60
    self.penalty_per_step = 100/self.real_steps
    self.initial_goal_distance = self.get_distance_to_goal()
    self.reward_per_meter = 1000/self.initial_goal_distance
    self.closest_to_goal = self.initial_goal_distance

    speed_mean = self.full_data['Speed_ov_ground'].iloc[self.start_indice+31:self.start_indice+61].mean()
    self.speed = math.ceil(speed_mean * 100.0) / 100.0
    self.total_reward = 0

    if self.viewer:
      self.viewer.close()
      self.viewer = None

    return self.get_current_observation()  # reward, done, info can't be included


  def render(self, mode='human'):
    if mode == 'console':
        current_state = self.get_current_state()
        print('Rudder: {}'.format(util.rescale(current_state.at['Rudder'], 'Rudder')))
        print('Speed: {}'.format(util.rescale(current_state.at['Speed_ov_ground'], 'Speed_ov_ground')))
        print('Diversion: {}'.format(self.get_diversion_from_real_route()))
        print('Distance to goal: {}'.format(self.get_distance_to_goal()))
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

        current_state = self.get_current_state()
        last_boat_lat = util.rescale(current_state.at['Latitude'], 'Latitude')
        last_boat_long = util.to_angle(current_state.at['Longitude_cos'], current_state.at['Longitude_sin'])
        self.last_boat_loc = (last_boat_long*ZOOM, last_boat_lat*ZOOM)
        real_boat_loc_array = self.get_real_boat_location()
        last_real_boat_lat = util.rescale(real_boat_loc_array[0], 'Latitude')
        last_real_boat_long = util.to_angle(real_boat_loc_array[1], real_boat_loc_array[2])
        self.last_real_boat_loc= (last_real_boat_long*ZOOM, last_real_boat_lat*ZOOM)

    real_boat_loc_array = self.get_real_boat_location()
    real_boat_heading = self.get_real_boat_heading_angle()
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

    current_state = self.get_current_state()
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
    print('Diversion: {}'.format(self.get_diversion_from_real_route()))
    print('Distance to goal: {}'.format(self.get_distance_to_goal()))

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')


  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

    '''
    dirpath = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dirpath, 'full_history_modelless.csv')
    self.full_history.to_csv(filepath, index=False)
    '''

    print('Time taken: {}'.format(time.process_time() - self.start))
    

  #added functions from state_estimator
  def get_diversion_from_real_route(self):
        return util.calculate_distance(self.get_real_boat_location(), 
                                       self.get_current_boat_location())

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

  def get_distance_to_goal(self):
      return util.calculate_distance(self.desired_goal, 
                                      self.get_current_boat_location())

  def get_current_boat_location(self):
        current_state = self.state_history.tail(1).iloc[0]
        return np.array([current_state.at['Latitude'], 
                         current_state.at['Longitude_cos'], 
                         current_state.at['Longitude_sin']])

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
      return self.get_current_observation_custom(self.observation_features)

  def get_next_observation(self, rudder_action):
    #calculate next state
    self.dict = {} #dictionary to hold features
    
    self.dict["Rudder"] = rudder_action #add rudder action
    self.dict["Speed_ov_ground"] = self.speed

    current_state = self.state_history.tail(1).iloc[0]
    # calculate next heading
    prev_angle = util.to_angle(current_state.at['Heading_ov_ground_cos'], current_state.at['Heading_ov_ground_sin'])
    new_angle = prev_angle + util.rescale(rudder_action, 'Rudder')
    self.dict['Heading_ov_ground_cos'], self.dict['Heading_ov_ground_sin'] = util.to_cos_sin(new_angle)

    # calculate next lat-long
    self.dict["Latitude"], self.dict["Longitude_cos"], self.dict["Longitude_sin"] = util.calculate_next_location(
        current_state.at['Latitude'], current_state.at['Longitude_cos'], current_state.at['Longitude_sin'], 
        current_state.at['Speed_ov_ground'], current_state.at['Heading_ov_ground_cos'], current_state.at['Heading_ov_ground_sin'],
    )

    #append next state
    next_state = pd.DataFrame(data=self.dict, index=[self.next_row])
    self.state_history = self.state_history.append(next_state)
    #self.full_history = self.full_history.append(next_state)

    #drop oldest row
    self.state_history.drop(self.state_history.head(1).index, inplace=True)

    self.next_row += 1

    return self.get_current_observation_custom(self.observation_features)

  def get_current_state(self):
      return self.state_history.tail(1).iloc[0]


