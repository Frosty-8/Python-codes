import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rich import print as rprint

data = np.random.randint(0,255,(10,5))
rprint('Original Data: ', data)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
rprint('\nScaled Data: ', scaled_data)