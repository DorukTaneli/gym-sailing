import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
import mlflow
import os
import random
import time
from gym_jtr.envs import util

class MemoryLeak:
    
    def __init__(self):
        self.boat_states = [
            'AWS', 
            'Yaw_cos', 'Yaw_sin',
            'Pitch', 'AWA_cos', 'AWA_sin',
#            'Roll', 'Heading_ov_ground_cos', 'Heading_ov_ground_sin',
#            'Speed_ov_ground', 'Heading_Mag_cos', 'Heading_Mag_sin',
#            'VMG', 'TWA_cos', 'TWA_sin',
#            'Speed_ov_surface', 'Heading_True_cos', 'Heading_True_sin',
            ]
        self._load_models()
        dirpath = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(dirpath, 'atlantic_usuable_1.csv')
        self.full_data = pd.read_csv(filepath)
        self.start_indice = random.randint(0, 23800)
        self.state_history = self.full_data.iloc[self.start_indice:self.start_indice+61]

    def _load_models(self):
        dirpath = os.path.dirname(os.path.abspath(__file__))
        self.models_dict = {}
        for target in self.boat_states:
            self.models_dict["{}model".format(target)] = mlflow.pyfunc.load_model(
                os.path.join(dirpath, 'models/{}/artifacts/{}model'.format(target, target)))

    def predict(self):
        for target in self.boat_states: #predict next boat states
            time_series = util.time_series_gen(data=self.state_history, target=target, batch_size=1, fs=1)
            #tensor = tf.convert_to_tensor(time_series, dtype=np.float32)
            np.squeeze(self.models_dict["{}model".format(target)].predict(time_series))
        K.clear_session()



leak = MemoryLeak()
start = time.process_time()
for x in range(10):
    leak.predict()
print('DONE')
print(time.process_time() - start)


    