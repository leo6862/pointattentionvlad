import sys 
import os 
import numpy as np
import shutil 
import pyflann
from tqdm import tqdm 
import open3d as o3d 
import threading

seqs = ["urban00","urban01","urban02","urban03","urban04","urban05","urban06","urban07","urban08","urban09",
        "urban10","urban11","urban12","urban13","urban14","urban15","urban16","urban17"]
data_root = "/home/yyj/catkin_ws/src/kaist_yyj/data"

for i,seq in enumerate(seqs):
    print("processing seq {} / {}".format(i,len(seqs)))
    seq_path = os.path.join(data_root,seq)
    pcd_files = os.listdir(seq_path)
    
    for pcd_file in tqdm(pcd_files,desc='Processing', unit='pcd'):
        output_dir = data_root + "/processed_" + seq + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = output_dir + pcd_file[:-4] + ".bin"
        pcd_path = os.path.join(seq_path,pcd_file) #生成完整的路径
        raw_cloud = o3d.io.read_point_cloud(pcd_path) 
        points_np = np.asarray(raw_cloud.points)
        flann = pyflann.FLANN()
        params = flann.build_index(points_np, algorithm='kdtree', trees=4)
        query_points = points_np
        res,dist = flann.nn_index(query_points, num_neighbors=16)
        processed_cloud = np.concatenate((points_np, res), axis=1) #n*19
        
        #将该文件保存为 outfile 
        with open(output_path, 'wb') as f:
            processed_cloud.tofile(f)


