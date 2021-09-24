import numpy as np
import pandas as pd
import math
import os
from gym_jtr.envs import util

class RomanReward:

    def __init__(self):
        self.polars = self.load_polars()

    def get_roman_reward(self, current_state, desired_goal):
        lamda = 0.8
        a = 1
        speed_pol = self.get_polar_speed(
            current_state.at['TWA_sin'],
            current_state.at['TWA_cos'],
            current_state.at['TWS'],
            current_state.at['Speed_ov_surface']
        )
        vmc = self.get_VMC(current_state, desired_goal)

        if current_state.at['Speed_ov_surface'] < speed_pol:
            x = np.array([
                current_state.at['Speed_ov_surface'],
                vmc,
                current_state.at['Pitch']
            ])
            x_ref = np.array([speed_pol, speed_pol, 0])
            LAMBDA = np.diag([1, 0.75, 1])
            reward = lamda*math.exp(
                -(1/a**2) *
                np.matmul(np.matmul(np.transpose(x-x_ref), LAMBDA), (x-x_ref))
            )
        else:
            x = np.array([vmc, current_state.at['Pitch']])
            LAMBDA = np.diag([1, 1])
            x_ref = np.array([
                current_state.at['Speed_ov_surface'],
                0
            ])
            reward = lamda*math.exp(
                -(1/a**2) * 1.48 *
                np.matmul(np.matmul(np.transpose(x-x_ref), LAMBDA), (x-x_ref))
            )

        return reward

    def get_VMC(self, current_state, desired_goal):
        R = 6378 * 1000
        way_p = (util.rescale(desired_goal[0], 'Latitude'),
                 util.rescale(util.to_angle(desired_goal[1], desired_goal[2]), 'Longitude')) 
        #way_p = (16.163504, - 61.521417+360)  # 'back of Guadeloupe' island
        lat_2 = current_state.at['Latitude']
        long_2 = util.to_angle(
            current_state.at['Longitude_cos'],
            current_state.at['Longitude_sin'],
        )
        heading = util.to_angle(
            current_state.at['Heading_True_cos'],
            current_state.at['Heading_True_sin'],
        )
        del_long = abs(way_p[1] - long_2)
        ang_cent = math.acos(
            math.sin(math.radians(lat_2)) * math.sin(math.radians(way_p[0])) +
            math.cos(math.radians(lat_2))*math.cos(math.radians(way_p[0])) *
            math.cos(math.radians(del_long))
        )

        big_angle = math.radians(180) -\
            math.asin(
                math.sin(math.radians(90-way_p[0])) *
                math.sin(math.radians(del_long)) /
                math.sin(ang_cent)
            )
        way_ang = big_angle-math.radians(heading-180)
        VMC = current_state.at['Speed_ov_surface'] * math.cos(way_ang)

        return VMC

    def load_polars(self):
        dirpath = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(dirpath, 'polars.csv')
        pp = pd.read_csv(filepath, index_col=0)
        return pp

    def get_polar_speed(self, TWA_sin, TWA_cos, TWS, scale):
        precise_ang = 0
        precise_speed = 0
        TWS = TWS/scale
        for i in range(0, self.polars.shape[1]-1):
            if TWS > int(self.polars.columns[i][0:2]) and\
               TWS < int(self.polars.columns[i+1][0:2]):
                speed = (
                    int(self.polars.columns[i][0:2]),
                    int(self.polars.columns[i+1][0:2])
                )
                col = i
                break
            elif TWS == int(self.polars.columns[i][0:2]):
                col = i
                precise_speed = 1
                break

        TWA = math.degrees(math.atan(TWA_sin/TWA_cos))

        if (TWA_sin < 0 and TWA_cos < 0) or (TWA_sin > 0 and TWA_cos < 0):
            TWA = TWA + 180
        elif TWA_sin < 0 and TWA_cos > 0:
            TWA = TWA + 360
        if TWA > 330:
            return 100
        for j in range(0, self.polars.shape[0]-1):
            if TWA > int(self.polars.index[j]) and\
               TWA < int(self.polars.index[j+1]):
                angle = (
                    int(self.polars.index[j]),
                    int(self.polars.index[j+1])
                )
                row = j
                break
            elif TWA == int(self.polars.index[j]):
                row = j
                precise_ang = 1
                break

        if precise_ang and precise_speed:
            pol_speed = self.polars.values[row, col]

        elif precise_ang and not precise_speed:
            m = (self.polars.values[row, col+1] -
                 self.polars.values[row, col]) / (speed[1]-speed[0])
            pol_speed = self.polars.values[row, col] + (TWS-speed[0])*m

        elif not precise_ang and precise_speed:
            m = (self.polars.values[row+1, col] -
                 self.polars.values[row, col]) / (angle[1]-angle[0])
            pol_speed = sp_1 = self.polars.values[row, col] + (TWA-angle[0])*m

        elif not precise_ang and not precise_speed:
            # the interpolation: vertical
            m_1 = (self.polars.values[row+1, col] -
                   self.polars.values[row, col]) / (angle[1]-angle[0])
            sp_1 = self.polars.values[row, col] + (TWA-angle[0])*m_1
            m_2 = (self.polars.values[row+1, col+1] -
                   self.polars.values[row, col+1]) / (angle[1]-angle[0])
            sp_2 = self.polars.values[row, col+1] + (TWA-angle[0])*m_2
            m_pol = (sp_2-sp_1)/(speed[1] - speed[0])
            pol_speed = sp_1 + m_pol * (TWS-speed[0])

        return pol_speed*scale
