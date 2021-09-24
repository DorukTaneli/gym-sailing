import pandas as pd
import numpy as np
import os

dirpath = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(dirpath, 'atlantic_usuable_1.csv')
full_data = pd.read_csv(filepath)

print(full_data.iloc[0:2])
print(full_data.head(2))
