# PointAttentionVLAD 

this is the official repo for paper `PointAttentionVLAD : A two-stage self-attenion network for point cloud based place recognition`

this code is mainly based on : https://github.com/cattaneod/PointNetVlad-Pytorch

Benchmark dataset Dowload address： https://drive.google.com/open?id=1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D

code for experiments on KAIST will be available soon. code for processing the raw KAIST　dataset to trainable data for neural network is based on c++， it wall be available soon.

## Install using docker 
docker is recommended for setting up the environment. All the scripts are recommended to run in the docker container.
```
    Pytorch 1.11 
    python 3.8 
```
Use following command to pull a pytorch1.11 docker image and generate a container 
```
docker pull pytorch/pytorch:1.11.0-cuda13-cudnn8-devel

docker run --gpus all --name torch_1.11  -it -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/${USERNAME}:/home/${USERNAME} -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e CONTAINER_NAME=torch_1.11  -e GDK_DPI_SCALE pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel /bin/bash
```

### dataset processing  
(N,3) point cloud (x\y\z)  to (N,19) point cloud (x\y\z\ind1\ind2\...\ind16) ind is the index of the nearest point of the point cloud.
This is for fast nearest neighbor search in the algorithm, no need to use a kd-tree for nn-search in the middle of the algorithm since nearest neighbor indices have been pre-calculated.
```
python dataset_preprocess.py
```


### Generate pickle files
```
cd generating_queries/

# For training tuples in our baseline network
python generate_training_tuples_baseline.py

# For training tuples in our refined network
python generate_training_tuples_refine.py

# For network evaluation
python generate_test_sets.py
```

### Train
```
python train_pointattentionvlad.py
```

### Evaluate
```
python eval_model.py
```

### PointAttentionVLAD ON KAIST DATASET
`kaist_yyj` is the ROS　project for preprossing KAIST　dataset.

Code for processing is based on C++ and ROS．　The reason why I use ROS for processing KAIST dataset is RVIZ is a good visualizer for point cloud. The reason why i use C++ is simply C++ is faster than python. And I use c++ multi-thread to improve program speed.

1. Download the KAIST DATASET @ https://sites.google.com/view/complex-urban-dataset/download-lidar#h.sa42osfdnwst
2. rename all the downloaded zip files to this format "urban00-info" & "urban00_data-001",then 
```
#if all your kasit daset zip files set is Download @ ~/Downloads , then YOURPATH = ~/Downloads
cp kaist_yyj/scripts/merge_files.sh YOUR_PATH 
cd YOUR_PATH 
./merge_files.sh
```
this step is used to unify the directory structure. For better kaist dataset usage.
3. run `kaist_yyj` package `run_submap_generator` node to use the VLP lidar in KAIST dataset to generate point cloud submaps.

4. (n,3) cloud to (n,19) cloud to skip online kd-tree based nearest neighbor search.
```
python kaist_yyj/run_submap_generator/scripts/generate_knn_data.py
```

5.  generating training & tesing tuples 
```
cd kaist_yyj/scripts/
python ./generating_training_tuple_v2.py
python ./generating_testing_tuple.py
```

6. start training model
```
python kaist_train.py 
```



### Pre-trained model Download 
1. pretrained model on Benchmark dataset 
https://drive.google.com/file/d/1XMLPiJpQBRLBN-ECdYHbcL_ADHNEDZT9/view?usp=drive_link

2. pretrained model on KAIST dataset 
https://drive.google.com/file/d/10ww7S1oWum6JYzt9Rz29e1T-FeEYtc_5/view?usp=drive_link

### LAST 
sorry guy , this repo is not that clean as you expected. The paper is online on May 6 2024. I am little buzy right now.
When i have time, i will clean this repo up.

