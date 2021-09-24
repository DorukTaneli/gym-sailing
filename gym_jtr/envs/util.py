import math
import haversine as hs
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

#in meters
def calculate_distance(goal_a, goal_b):
    #print('calculate_distance({},{})'.format(goal_a, goal_b))
    #np.savetxt('/home/tane/jtr/goal_a.txt', goal_a, delimiter=',')  
    assert goal_a.shape == goal_b.shape
    loc_a = (rescale(goal_a[0], 'Latitude'), to_angle(goal_a[1], goal_a[2])) #get lat(0), long_cos(1), long_sin(2)
    loc_b = (rescale(goal_b[0], 'Latitude'), to_angle(goal_b[1], goal_b[2]))
    return hs.haversine(loc_a, loc_b, unit='m')

#from https://gist.github.com/jeromer/2005586
def calculate_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing
    
def to_angle(cos, sin):
    #print('to_angle({},{})'.format(cos, sin))
    return math.degrees(math.atan2(sin, cos))

def to_cos_sin(angle):
    cos = math.cos(math.radians(angle))
    sin = math.sin(math.radians(angle))
    return cos, sin

def calculate_next_location(lat, long_cos, long_sin, speed_ov_ground, heading_ov_ground_cos, heading_ov_ground_sin):

    #upscale for calculations
    lat = rescale(lat, 'Latitude')
    speed_ov_ground = rescale(speed_ov_ground, 'Speed_ov_ground')

    R = 6378000  # m the Earth radius
    lat1 = math.radians(lat)
    long1 = math.radians(to_angle(long_cos, long_sin))
    speed = speed_ov_ground*0.5144447  # convert knots to m/s
    head = math.radians(to_angle(heading_ov_ground_cos, heading_ov_ground_sin))

    lat2 = math.asin(
        math.sin(lat1) * math.cos(speed/R) +
        math.cos(lat1) * math.sin(speed/R) * math.cos(head)
    )
    dlon = math.atan2(
        math.sin(head) * math.sin(speed/R)*math.cos(lat1),
        math.cos(speed/R)-math.sin(lat1)*math.sin(lat2)
    )
    long2 = long1+dlon

    latitude = scale(math.degrees(lat2), 'Latitude')
    longitude_cos, longitude_sin = to_cos_sin(math.degrees(long2))

    return latitude, longitude_cos, longitude_sin

def time_series_gen(data, target, batch_size, fs):
    ''' Creates batches of temporal data from data. '''
    input_len = 60 # 60 seconds
    length = fs*input_len # number of timesteps
    np_data = data.to_numpy()
    np_targets = data[target].to_numpy()
    return TimeseriesGenerator(data=np_data, targets=np_targets, length=length, batch_size=batch_size)

def min_max_norm(x, target):
    ''' For features contained between 0 and strictly positive value '''
    
    min_max_features = ['AWS', 'Air_temp', 'Current_speed', 'Speed_ov_ground',
                        'VMG', 'Speed_ov_surface', 'TWS', 'Heading_ov_ground']
    
    min_max_bounds = {'AWS': [0,50],
                     'Air_temp': [0,30],
                      'Current_speed': [0,15],
                      'Speed_ov_ground': [0,25],
                      'VMG': [0,25],
                      'Speed_ov_surface': [0,25],
                      'TWS': [0,40],
                      'Heading_ov_ground': [0,360]
                     }
    
    if target in min_max_features:
        min_ = min_max_bounds[target][0]
        max_ = min_max_bounds[target][1]
        x = (x-min_)/(max_-min_)
        
    return x
    
def max_abs_norm(x, target):
    ''' For features contained in symmetric interval'''
    
    max_abs_features = ['Latitude', 'Pitch', 'Roll', 'Rudder', 'Longitude']

    max_abs_bounds = {'Latitude': [-90, 90],
                      'Pitch': [-180, 180],
                      'Roll': [-60, 60],
                      'Rudder': [-50, 50],
                      'Longitude': [-180, 180]
                     }
    
    if target in max_abs_features:
        max_ = max_abs_bounds[target][1]
        x = x/max_
        
    return x

def scale(x, target):
    ''' Apply both forms of scaling. '''
    x = min_max_norm(x, target)
    x = max_abs_norm(x, target)
    return x

def rescale(y, target):
    ''' Rescale y to original scale. '''
    
    min_max_bounds = {'AWS': [0,50],
                 'Air_temp': [0,30],
                  'Current_speed': [0,15],
                  'Speed_ov_ground': [0,25],
                  'VMG': [0,25],
                  'Speed_ov_surface': [0,25],
                  'TWS': [0,40],
                  'Heading_ov_ground': [0,360]}

    max_abs_bounds = {'Latitude': [-90, 90],
                  'Pitch': [-180, 180],
                  'Roll': [-60, 60],
                  'Rudder': [-50, 50],
                  'Longitude': [-180, 180]}
    
    if target in list(min_max_bounds.keys()):
        min_ = min_max_bounds[target][0]
        max_ = min_max_bounds[target][1]
        y = y*(max_-min_) + min_
    
    elif target in list(max_abs_bounds.keys()):
        max_ = max_abs_bounds[target][1]
        y = y*max_
    
    return y