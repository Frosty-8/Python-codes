import numpy as np
import matplotlib.pyplot as plt

data_group1 = np.random.normal(loc=50,scale=10,size=100)
data_group2 = np.random.normal(loc=200,scale=15,size=100)

plt.hist(data_group1,bins=20,label='Group 1',color='red')
plt.hist(data_group2,bins=20,label='Group 2',color='green')
plt.legend()
plt.show()  