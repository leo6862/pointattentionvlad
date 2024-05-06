import numpy as np
import os

filename = "/home/yyj/download/kaist/urban17/sensor_data/VLP_left/1524211202189730000.bin";
pc = np.fromfile(filename, dtype=np.float64)

# if(pc.shape[0] != 4096*19):
#     print("Error in pointcloud shape")
#     return np.array([])
# pc = np.reshape(pc,(pc.shape[0]//19, 19))
# return pc
pc = pc.reshape((-1,4))
print(pc.shape[0])