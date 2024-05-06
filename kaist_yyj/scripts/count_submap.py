import os 
import sys 

train_seqs = ["urban01","urban02","urban03","urban05","urban06","urban07","urban08",
        "urban10","urban11","urban12","urban13","urban15","urban16","urban17"]

test_seqs = ["urban00","urban04","urban09","urban14"]

train_file_count = 0
test_file_count = 0

data_root = "/home/yyj/catkin_ws/src/kaist_yyj/data"

for seq in train_seqs:
    dir_path = os.path.join(data_root,seq) 
    file_names = os.listdir(dir_path)
    train_file_count += len(file_names)

print("train file : {}".format(train_file_count))

for seq in test_seqs:
    dir_path = os.path.join(data_root,seq) 
    file_names = os.listdir(dir_path)
    test_file_count += len(file_names)

print("test file : {}".format(test_file_count))

    

