import numpy as np
from pathlib import Path
import sys
import os
import pickle
import math 
from tqdm import tqdm

"""
原始的 training tuple 
  1.读取当前的 txt 文件 读取 时间戳 & pos 
  2.将 dir 中的 点云文件全部读取出来 ， 将 使用 名字(时间戳) & 
  3.对于每个点云 ， 寻找 欧氏距离绝对值 在 x 之内的所有点云 ， 计算 距离 大于 y 的所有点云组成 neg tuple . 
  4. 生成一个 dict 用于存储当前的 pos & neg tuple. 
"""

"""
仿照 PointNetVLAD 中的 pickle 生成方式
dict = {0 : {"query" : 文件路径 ('oxford/2014-05-19-13-20-57/pointcloud_20m_10overlap/1400505894395159.bin') 
             , 'positives' : [] , 'negatives' : [] }  }  # 没有other neg : other neg 是在 训练的脚本中生成的
             
1. 将所有的 seq 的 stamp 都读到内存中 [(stamp , pose)]
2. 初始化字典 queries = {0...n : {'query' : [] , 'positives' : [] , 'negatives' : []}}  像上文中的 dict 
3. 因为每个seq 中的submap数量不多所以直接 一个个暴力的比较就行
4. 保存 dict
"""


seqs = ["urban00","urban04","urban09","urban14"]

data_root = "/home/yyj/catkin_ws/src/kaist_yyj/data"
pos_dist_thre = 30
neg_dict_thre = 100




def euclidean_pose(pos1,pos2):
    x_diff = pos1[0] - pos2[0]
    y_diff = pos1[1] - pos2[1]
    z_diff = pos1[2] - pos2[2]
    return math.sqrt(x_diff * x_diff + y_diff * y_diff +z_diff * z_diff)

def check_processed_bin_file_exists(seq,stamp):
    processed_file_name = "processed_" + seq 
    processed_file_dir = os.path.join(data_root,processed_file_name)
    
    #判断当前文件夹是否存在
    if not os.path.exists(processed_file_dir):
        return False
    
    #读取当前文件夹中的所有内容 ， 
    files = os.listdir(processed_file_dir)
    stamps_ = [float(file[:-4]) for file in files]
    if stamp not in stamps_:
        return False 
    return True 


def check_data_exist(data_root,seq):
    txt_file_name = seq + ".txt"
    dir_path = os.path.join(data_root,seq)
    txt_path = os.path.join(data_root,txt_file_name)
    if os.path.exists(dir_path):
        pass
    else:
        print(dir_path + " not exist")
        return False 

    if os.path.exists(txt_path):
        pass
    else:
        print(txt_path + " not exist")
        return False 
    return True

def check_data_complete(data_root,seq):
    """
        1.check_data_exist
        2. load txt file  , get stamp -> pose dict
        3. load file name in dir , get stamp list 
        4. check if every file in txt file
    """
    if not check_data_exist(data_root,seq):
        return False
    
    txt_dict = {} #stamp -> pos dict

    txt_file_name = seq + ".txt"
    dir_path = os.path.join(data_root,seq)
    txt_path = os.path.join(data_root,txt_file_name)
    #TODO Load txt file
    with open(txt_path,'r') as txt_file:
        for line in txt_file:
            values = line.split()  # 默认使用空格拆分每行的字段
            if len(values) != 4:
                print("load " + txt_path + " failed . Not 4 float num in row , data = ",[float(value) for value in values])
                return False
            else:
                float_values = [float(value) for value in values]
                stamp = float_values[0]
                x,y,z = float_values[1],float_values[2],float_values[3]
                txt_dict[stamp] = [x,y,z]
    
    files = os.listdir(dir_path)
    for file_name in files:
        stamp_str = file_name[:-4]
        # print("stamp_str: " , stamp_str)
        stamp = float(stamp_str)
        if stamp not in txt_dict:
            print(stamp_str + " not in txt file.error.")
            return False
    return True

for seq in seqs:
    seq_dict = {} #global_index -> {"query:" : []  , "positive" : [],"negtive":[]}
    if not check_data_complete(data_root,seq):
        print("seq : " + seq + " not complete. skip")
        continue
    print(seq + " data perfect . let's go!")
    id = 0
    txt_dict = {} #stamp -> pos dict 
    txt_file_name = seq + ".txt" 
    dir_path = os.path.join(data_root,seq) 
    txt_path = os.path.join(data_root,txt_file_name)
    #Load txt file
    with open(txt_path,'r') as txt_file:#读取 stamp & pose txt
        for line in txt_file:
            values = line.split()  # 默认使用空格拆分每行的字段
            float_values = [float(value) for value in values]
            stamp = float_values[0]
            x,y,z = float_values[1],float_values[2],float_values[3]
            #检查 processed 目录下是否存在 当前的 stamp.bin文件
            if not check_processed_bin_file_exists(seq,stamp):
                sys.exit()
            txt_dict[stamp] = [x,y,z,id] #生成 stmap -> pose * global_id 
            id += 1
    #for every pos in txt dict 
    for q_stamp,q_pos_ind in tqdm(txt_dict.items(),desc='Processing', unit='current submap num'):
        q_pos = q_pos_ind[:3]
        q_ind = int(q_pos_ind[3])
        query_path = "processed_" + seq + "/" + str('%.6f'% q_stamp) + ".bin"
        seq_dict[q_ind] = {"query":query_path,"positives":[],"negatives":[]}
        print("query_path = " + query_path)
        
        for i_stamp ,i_pos_ind in txt_dict.items():
            i_pos = i_pos_ind[:3]
            i_ind = int(i_pos_ind[3])
            if q_ind == i_ind:
                continue
            dist = euclidean_pose(q_pos,i_pos)
            if dist < pos_dist_thre:
                seq_dict[q_ind]["positives"].append(i_ind)
            elif dist > neg_dict_thre:
                seq_dict[q_ind]["negatives"].append(i_ind)
        print("{} submap pos_num : {}   neg_num: {}".format(q_ind,len(seq_dict[q_ind]["positives"]),len(seq_dict[q_ind]["negatives"])))
        print(seq + " data done!")
    pkl_file_name = seq + "_testing_tuple.pickle"
    pickle_file_name = os.path.join(data_root,pkl_file_name)
    with open(pickle_file_name, 'wb') as handle:
        pickle.dump(seq_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
