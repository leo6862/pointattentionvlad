import sys 
import os 
import numpy as np
import shutil 
import pyflann
from tqdm import tqdm 
import threading

#bench_mark data set path 
base_dir = "/home/yyj/dl_dataset/benchmark_datasets/"

#oxford dataset path 
runs_folder = "oxford/"

#oxforad files 
non_overlap_file_name = "/pointcloud_20m/"
non_overlap_csv_name = "/pointcloud_locations_20m.csv"
overlap_file_name = "/pointcloud_20m_10overlap/"
overlap_csv_name = "/pointcloud_locations_20m_10overlap.csv"

#inhouse & university files 
inhouse_25m_10_file_name = "/pointcloud_25m_10/"
inhouse_25m_25_file_name = "/pointcloud_25m_25/"
inhouse_10_csv_name = "/pointcloud_centroids_10.csv"
inhouse_25_csv_name = "/pointcloud_centroids_25.csv"

output_data_set_path = "/home/yyj/dl_dataset/nn_neighbor_benchmark_dataset/"
if not os.path.exists(output_data_set_path):
    os.makedirs(output_data_set_path)
output_data_set_path = output_data_set_path + runs_folder
if not os.path.exists(output_data_set_path):
    os.makedirs(output_data_set_path)

all_folders=sorted(os.listdir(os.path.join(base_dir,runs_folder)))

def load_pc_file(filename):
	#returns Nx3 matrix
	pc=np.fromfile(filename, dtype=np.float64)

	if(pc.shape[0]!= 4096*3):
		print("Error in pointcloud shape")
		return np.array([])

	pc=np.reshape(pc,(pc.shape[0]//3,3))
	return pc

def load_processed_pc_file(filename):
	#returns Nx3 matrix
	pc=np.fromfile(filename, dtype=np.float64)

	if(pc.shape[0]!= 4096*19):
		print("Error in pointcloud shape")
		return np.array([])

	pc=np.reshape(pc,(pc.shape[0]//19,19))
	return pc

def copy_and_process_cloud_file(input_file,output_file):

    all_bin_file = os.listdir(input_file)
    
    if not os.path.exists(output_file):
        os.mkdir(output_file)
    #对于当前的每个bin file 
    for bin_file in all_bin_file:
        current_bin_file_path = input_file + bin_file
        output_bin_file_path = output_file + bin_file
        raw_cloud = load_pc_file(current_bin_file_path)
        # 将当前raw_cloud加入kd tree 中
        flann = pyflann.FLANN()
        params = flann.build_index(raw_cloud, algorithm='kdtree', trees=4)
        query_points = raw_cloud
        res,dist = flann.nn_index(query_points, num_neighbors=16)
        processed_cloud = np.concatenate((raw_cloud, res), axis=1) #n*19
        
        #将该文件保存为 outfile 
        with open(output_bin_file_path, 'wb') as f:
            processed_cloud.tofile(f)



def ops_in_one_folder(folder):
    current_input_folder_path = base_dir + runs_folder + folder
    current_output_folder_path = output_data_set_path + folder
    if not os.path.exists(current_output_folder_path):
        os.mkdir(current_output_folder_path)
    #oxford 
    current_in_overlap_file_name = current_input_folder_path+overlap_file_name
    current_in_overlap_csv_name = current_input_folder_path + overlap_csv_name
    current_in_non_overlap_file_name = current_input_folder_path + non_overlap_file_name
    current_in_non_overlap_csv_name = current_input_folder_path + non_overlap_csv_name
    
    #inhouse 
    current_in_inhouse_25_file_name = current_input_folder_path + inhouse_25m_25_file_name
    current_in_inhouse_10_file_name = current_input_folder_path + inhouse_25m_10_file_name
    current_in_inhouse_25_csv_name =  current_input_folder_path + inhouse_25_csv_name
    current_in_inhouse_10_csv_name =  current_input_folder_path + inhouse_10_csv_name
    
    # print('current_output_folder_path+overlap_file_name = {}'.format(current_output_folder_path+overlap_file_name))
    #当前in文件夹下存在pointcloud_20m_10overlap (每个文件夹下都有)
    #! 以下是oxaford 中的内容
    if os.path.exists(current_in_overlap_file_name):
        copy_and_process_cloud_file(current_in_overlap_file_name,current_output_folder_path+overlap_file_name)
    if os.path.exists(current_in_non_overlap_file_name):
        copy_and_process_cloud_file(current_in_non_overlap_file_name,current_output_folder_path+non_overlap_file_name)
    if os.path.exists(current_in_overlap_csv_name):
        shutil.copy(current_in_overlap_csv_name,current_output_folder_path)
    if os.path.exists(current_in_non_overlap_csv_name):
        shutil.copy(current_in_non_overlap_csv_name,current_output_folder_path)
        
    # 以下是 bussiness & university 中的内容
    if os.path.exists(current_in_inhouse_25_file_name):
        copy_and_process_cloud_file(current_in_inhouse_25_file_name,current_output_folder_path+inhouse_25m_25_file_name) #TODO 改名称
    if os.path.exists(current_in_inhouse_10_file_name):
        copy_and_process_cloud_file(current_in_inhouse_10_file_name,current_output_folder_path+inhouse_25m_10_file_name) #TODO 改名称
    if os.path.exists(current_in_inhouse_25_csv_name):
        shutil.copy(current_in_inhouse_25_csv_name,current_output_folder_path)
    if os.path.exists(current_in_inhouse_10_csv_name):
        shutil.copy(current_in_inhouse_10_csv_name,current_output_folder_path)
    print('{} done.'.format(folder))
    # print('---')

#! 注意有的文件夹内可能有四个文件pointcloud_20m            pointcloud_locations_20m_10overlap.csv
#!                            pointcloud_20m_10overlap  pointcloud_locations_20m.csv
for folder in all_folders:
    t = threading.Thread(target=ops_in_one_folder,args=(folder,))
    t.start()
    
    
