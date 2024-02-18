import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle

global_path = sys.argv[1]
trajectory_dict = sys.argv[2]

global_array = np.load(global_path)
trajectory_dict = pickle.load(open(trajectory_dict, 'rb'))

plt.plot(global_array[:,0], global_array[:,1], 'r')

for t in trajectory_dict.keys():
    plt.plot(trajectory_dict[t][:,0], trajectory_dict[t][:,1], 'b')
    

plt.show()