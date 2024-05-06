#!/bin/bash

seqs=("urban00" "urban01" "urban02" "urban03" "urban04" "urban05" "urban06" "urban07" "urban08" "urban09" "urban10" "urban11" "urban12" "urban13" "urban14" "urban15")
info_dir_suffix="-info"
data_dir_suffix="_data-001"
slash="/"
calibrartion_dir_name="calibration"
mid_calib_dir_suffix="_calibration"
mid_pose_dir_suffix="_pose"
globa_pose_file_name="global_pose.csv"
sensor_data_str="sensor_data"
for seq in "${seqs[@]}"
do 
    mkdir "$seq" -p 
    seq_info_dir="$seq$info_dir_suffix"
    seq_data_dir="$seq$data_dir_suffix"
    mid_calib_dir_name="$seq$mid_calib_dir_suffix"

    #                 urban16-info  /     urban16 / urban16_calibration /   urban16/ calibration
    calibration_dir="$seq_info_dir$slash$seq$slash$mid_calib_dir_name$slash$seq$slash$calibrartion_dir_name"
    cp $calibration_dir "$seq$slash" -r  #move calibration dir 
    
    mid_pose_dir_name="$seq$mid_pose_dir_suffix"
    globa_pose_csv_path="$seq_info_dir$slash$seq$slash$mid_pose_dir_name$slash$seq$slash$globa_pose_file_name"
    cp $globa_pose_csv_path "$seq$slash" #move global_pose.csv 

    sensor_data_dir="$seq_data_dir$slash$seq$slash$sensor_data_str"
    mv $sensor_data_dir "$seq$slash" #move sensor_data
done