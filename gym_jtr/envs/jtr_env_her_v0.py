#default imports
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

#added imports
from gym_jtr.envs import state_estimator
from gym_jtr.envs import util
import pandas as pd
import os
import time
import math

WINDOW_W = 500
WINDOW_H = 500
ZOOM = 1200

class JtrEnvHERV0(gym.GoalEnv):

  """
  A goal-based environment. It functions just as any regular OpenAI Gym environment but it
  imposes a required structure on the observation_space. More concretely, the observation
  space is required to contain at least three elements, namely `observation`, `desired_goal`, and
  `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
  `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
  actual observations of the environment as per usual.
  """

  metadata = {'render.modes': ['human']}

  def __init__(self):
    print('init called')
    super(JtrEnvHERV0, self).__init__()

    self.start = time.process_time()
    self.viewer = None
    self.state_estimator = state_estimator.StateEstimator(her=True)

    # goal_distance_threshold (float): the threshold after which a goal is considered achieved
    self.goal_distance_threshold = 10 
    # reset_distance_threshold (float): the threshold of deviation from main route after which the environment is reset
    self.reset_distance_threshold = 10
    # 'sparse' or 'dense'
    self.reward_type = 'sparse'

    # action space - Rudder
    self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) #scaled between -50, 50

    # observation space - environment
    # all features are scaled to between -1 and 1
    self.observation_space = spaces.Dict({
      'desired_goal': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32), #lat, long_cos, long_sin
      'achieved_goal': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32), #lat, long_cos, long_sin    
      'observation': spaces.Box(low=-1, high=1, shape=(29,), dtype=np.float32), #all 29 features
    })

  def step(self, action):
    print('step called')
    observation = self.state_estimator.get_next_observation(action)

    done = False
    if (self.state_estimator.get_diversion_from_real_route() > self.reset_distance_threshold or
        self.state_estimator.get_distance_to_goal() < self.goal_distance_threshold):
      done = True

    info = {} #not used

    reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)

    print('step {}, reward: {}, diversion: {}'.format(self.state_estimator.next_row - (self.state_estimator.start_indice + 61),
                                            reward,
                                            self.state_estimator.get_diversion_from_real_route()))

    return observation, reward, done, info

  def reset(self):
    print('reset called')
    # Enforce that each GoalEnv uses a Goal-compatible observation space.
    if not isinstance(self.observation_space, gym.spaces.Dict):
      raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')
    for key in ['observation', 'achieved_goal', 'desired_goal']:
      if key not in self.observation_space.spaces:
        raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))
    
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
    print('close called')
    if self.viewer:
      self.viewer.close()
      self.viewer = None
    print('Time taken: {}'.format(time.process_time() - self.start))
    
  
  def compute_reward(self, achieved_goal, desired_goal, info):
    """Compute the step reward. This externalizes the reward function and makes
    it dependent on an a desired goal and the one that was achieved. If you wish to include
    additional rewards that are independent of the goal, you can include the necessary values
    to derive it in info and compute it accordingly.
    Args:
        achieved_goal (object): the goal that was achieved during execution
        desired_goal (object): the desired goal that we asked the agent to attempt to achieve
        info (dict): an info dictionary with additional information
    Returns:
        float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
        goal. Note that the following should always hold true:
            ob, reward, done, info = env.step()
            assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
    """
    print('compute_reward:')
    print('achieved_goal: {}'.format(achieved_goal))
    print('desired_goal: {}'.format(desired_goal))
    distance = util.calculate_distance(achieved_goal, desired_goal)
    if self.reward_type == 'sparse':
      return -float(distance > self.goal_distance_threshold)
    else:
      return -distance
