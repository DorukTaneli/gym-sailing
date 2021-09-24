#default gym imports
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

#added imports
from gym_jtr.envs import util
from gym_jtr.envs.state_estimator_suggested import StateEstimatorSuggested
import time
import math

WINDOW_W = 500
WINDOW_H = 500
ZOOM = 1200

class JtrEnvSuggestedV0(gym.Env):

  metadata = {'render.modes': ['human', 'console']}

  def __init__(self):
    super(JtrEnvSuggestedV0, self).__init__()

    # initialize the state estimator
    self.state_estimator = StateEstimatorSuggested()

    #viewer for rendering
    self.viewer = None

    # length of Concise 8 in meters
    boat_length = 12 

    # the threshold after which a goal is considered achieved
    self.goal_distance_threshold = 3*boat_length

    # threshold multiplier for deviation from main route after which the environment is reset
    # reset_distance_threshold will be calculated later, as the goal_distance varies each run
    self.reset_distance_threshold_multiplier = 1/5 #reset if deviated more than 20% of total distance 

    # the timesteps to wait after real boat reaches location before the environment is reset
    # reset_timestep_threshold will be calculated later, as the runtime_minutes varies each run
    self.reset_timestep_threshold_multiplier = 1/4 #reset if it takes 25% longer

    # action space - Rudder
    # scaled down, original between -50, 50
    self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) 

    # all features are scaled to between -1 and 1
    # modify the features as needed, this is the suggested obs_space based on sailing knowledge
    # Speed_ov_ground, AWS, AWA, Pitch, Roll, Heading_ov_ground, heading to goal
    self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    # start timer to see how long training takes
    self.start = time.process_time()

  def step(self, action):
    observation = self.state_estimator.get_next_observation(action)
    self.step_no += 1

    reward = 0.0
    done = False
    if (self.state_estimator.get_diversion_from_real_route() > self.reset_distance_threshold or
        self.step_no > self.real_steps + self.reset_timestep_threshold):
      done = True
      reward -= 50
    elif self.state_estimator.get_distance_to_goal() < self.goal_distance_threshold:
      done = True
      reward += self.reward_per_meter * self.prev_distance_to_goal
    else:
      current_distance_to_goal = self.state_estimator.get_distance_to_goal()
      reward += self.reward_per_meter * (self.prev_distance_to_goal - current_distance_to_goal)
      self.prev_distance_to_goal = current_distance_to_goal

    reward -= self.penalty_per_step

    info = {} #not used

    '''
    # debug prints if needed
    if done:
      diversion = self.state_estimator.get_diversion_from_real_route()
      goal_distance = self.state_estimator.get_distance_to_goal()
      print('step {}, total_reward: {}'.format(self.step_no, self.total_reward))
      print('diversion: {}, distance to goal: {}'.format(diversion, goal_distance))
      print("last observation: {}".format(observation))
    '''
    
    return observation, reward, done, info


  def reset(self):
    self.state_estimator.reset()
    self.step_no = 0

    # variables for reward calculation of current episode
    real_steps = self.state_estimator.runtime_minutes*60
    self.penalty_per_step = 100/real_steps
    self.initial_goal_distance = self.state_estimator.get_distance_to_goal()
    self.reward_per_meter = 200/self.initial_goal_distance
    self.prev_distance_to_goal = self.initial_goal_distance
    self.reset_distance_threshold = self.initial_goal_distance*self.reset_distance_threshold_multiplier
    self.reset_timestep_threshold = real_steps*self.reset_timestep_threshold_multiplier

    raise NotImplementedError
    self.goal_lat_long = None #TODO: Calculate goal_lat_long -> tuple(lat, long)

    if self.viewer:
      self.viewer.close()
      self.viewer = None

    return self.state_estimator.get_current_observation()  # reward, done, info can't be included

  # Use 'console' mode instead of 'human' on Azure, to alleviate lack of display errors
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

        goal = rendering.make_circle(radius=0.25)
        goal.set_color(1, 0.9, 0.9)
        self.goal_transform = rendering.Transform()
        goal.add_attr(self.goal_transform)
        self.viewer.add_geom(goal)
        goal_lat = util.rescale(self.state_estimator.desired_goal[0], 'Latitude')
        goal_long = util.to_angle(self.state_estimator.desired_goal[1], self.state_estimator.desired_goal[2])
        self.goal_transform.set_translation(goal_long*ZOOM, goal_lat*ZOOM)

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

    current_lat_long = (util.rescale(current_state.at['Latitude'], 'Latitude'),
                        util.to_angle(current_state.at['Longitude_cos'], current_state.at['Longitude_sin']))
    goal_heading = util.calculate_compass_bearing(current_lat_long, self.goal_lat_long)
    print('====================================')
    print('Heading: {}'.format(heading))
    print('Goal_heading: {}'.format(goal_heading))
    print('====================================')

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')


  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

    print('Time taken: {}'.format(time.process_time() - self.start))
  


