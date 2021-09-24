import pyglet
import numpy as np

WINDOW_W = 800
WINDOW_H = 800

class RenderBoat:
    def __init__(self):
        self.viewer = None
        self.i = 0

    
    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            
            boat = rendering.make_polygon(v=[[-1, -1], [-1, 1], 
                                            [0, 2], [1, 1], [1, -1]], 
                                          filled=True)
            boat.set_color(0.9, 0.9, 1.0)
            self.boat_transform = rendering.Transform()
            boat.add_attr(self.boat_transform)
            self.viewer.add_geom(boat)

        self.boat_transform.set_rotation(self.i)
        self.i+=0.01


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


r = RenderBoat()

for i in range(100):
    r.render()