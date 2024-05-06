import open3d as o3d
import numpy as np
import sys

# 创建点云数据
pcd_file = "/home/yyj/catkin_ws/src/kaist_yyj/data/urban09/1523432469.337098.pcd"
if len(sys.argv) < 1:
    print("usage : load_pcd_o3d.py pcd_path")
pcd_file = sys.argv[1]

point_cloud = o3d.io.read_point_cloud(pcd_file) 

visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

visualizer.add_geometry(point_cloud)

visualizer.run()
visualizer.destroy_window()
