#ifndef KAIST_YYJ_KAIST_VLP_RECEIVER_H
#define KAIST_YYJ_KAIST_VLP_RECEIVER_H

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <iomanip>
#include <Eigen/Dense>
#include <glog/logging.h>

#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

using namespace std;
using namespace Eigen;

class Kaist_vlp_reciever {
 public:
  //直接输入 global_pose txt 的路径 ， 读取文件
  Kaist_vlp_reciever(ros::NodeHandle& _nh, const std::string _global_pose_txt_path) {
    nh = _nh;
    //load _global_pose_txt_path
    if (!init_pose_txt(_global_pose_txt_path))  {
      LOG(ERROR) << "init_pose_txt failed. check the global_pose_file_path!";
    }
    LOG(INFO) << "global pose file loaded.";
    init_calib();
    init_sub();
  }
 private:
  void init_sub() {
    sub_vlp_left  = nh.subscribe("/ns2/velodyne_points",10000,&Kaist_vlp_reciever::left_vlp_callback,this);
    sub_vlp_right = nh.subscribe("/ns1/velodyne_points",10000,&Kaist_vlp_reciever::right_vlp_callback,this);
    pub_right_velo = nh.advertise<sensor_msgs::PointCloud2>("/right_velo_imu",1000);
    pub_left_velo = nh.advertise<sensor_msgs::PointCloud2>("/left_velo_imu",1000);
    pub_right_velo_glo = nh.advertise<sensor_msgs::PointCloud2>("/right_velo_glo",1000);
    pub_left_velo_glo = nh.advertise<sensor_msgs::PointCloud2>("/left_velo_glo",1000);
  }
  void init_calib() {
    //这里的参数是参考的 /home/yyj/download/kaist/urban17/calibration
    Vehicle2RightVLP << -0.512152, 0.699241 , -0.498761,-0.449885 ,
                       -0.494811, -0.714859 ,-0.494104 ,-0.416713 ,
                       -0.702041, -0.0062642, 0.712109 , 1.91294  ,
                       0        ,    0      ,    0     ,    1     ;
    Vehicle2LeftVLP << -0.514066 , -0.702201 ,-0.492595,-0.449885 ,
                        0.486485 , -0.711672 ,0.506809 ,-0.416713 ,
                        -0.706447, 0.0208933 ,0.707457 ,  1.91294 ,
                        0        ,    0      ,   0     ,  1;
    Vehicle2IMU     << 1, 0, 0, -0.07, 
                       0, 1, 0,   0  ,
                       0, 0, 1,  1.7 ,
                       0, 0, 0,   1;
  }
  bool init_pose_txt( const std::string _global_pose_txt_path) {
    ifstream pose_file(_global_pose_txt_path);
    if (!pose_file.is_open()) {
        return false;
    }
    string line;
    while (getline(pose_file, line)) {
        std::vector<double> row;
        std::istringstream iss(line);
        string token;
        while (getline(iss, token, ',')) {
            row.push_back(stod(token));
        }
        double stamp = row[0] / 1e9;
        Matrix4d pose;
        pose << row[1] , row[2] , row[3] , row[4],
                row[5] , row[6] , row[7] , row[8],
                row[9] , row[10], row[11], row[12],
                0      ,       0,       0,   1;
        // LOG(INFO) << setprecision(18) << "stamp = " << stamp;
        // LOG(INFO) << "pose = \n" << pose;
        stamp_pose_map.insert(std::pair<double,Matrix4d>(stamp,pose));
    }
  }
  void left_vlp_callback(const sensor_msgs::PointCloud2ConstPtr& msg_in) {
    /*
      transform point cloud into imu frame , then seach pose in global_pose_map 
    */
    Eigen::Matrix4d left_vlp_2_imu = Vehicle2IMU.inverse() * Vehicle2LeftVLP;
    // Eigen::Matrix4d left_vlp_2_imu = Vehicle2LeftVLP.inverse() * Vehicle2IMU;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg_in,*cloud_in);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_imu(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_glo(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_in,*cloud_imu,left_vlp_2_imu);
    
    //找到当前时刻点云对应的位置
    double stamp = msg_in->header.stamp.toSec();
    Eigen::Matrix4f current_pose;
    if (!get_pose_from_stamp(stamp,current_pose)) {
      LOG(WARNING) << "this cloud msg pose not find.skip.";
      return;
    }
    pcl::transformPointCloud(*cloud_imu,*cloud_glo,current_pose);
    LOG(INFO) << "current pose : \n" << current_pose;
    //pub cloud imu 
    sensor_msgs::PointCloud2 cloud_imu_msg;
    pcl::toROSMsg(*cloud_imu,cloud_imu_msg);
    cloud_imu_msg.header.stamp = msg_in->header.stamp;
    cloud_imu_msg.header.frame_id = "imu";
    pub_left_velo.publish(cloud_imu_msg);

    //pub cloud global 
    sensor_msgs::PointCloud2 cloud_glo_msg;
    pcl::toROSMsg(*cloud_glo,cloud_glo_msg);
    cloud_glo_msg.header.stamp = msg_in->header.stamp;
    cloud_glo_msg.header.frame_id = "global";
    pub_left_velo_glo.publish(cloud_glo_msg);
 }

  void right_vlp_callback(const sensor_msgs::PointCloud2ConstPtr& msg_in) {
    Eigen::Matrix4d right_vlp_2_imu = Vehicle2IMU.inverse() * Vehicle2RightVLP;
    // Eigen::Matrix4d right_vlp_2_imu = Vehicle2RightVLP.inverse() * Vehicle2IMU;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg_in,*cloud_in);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_imu(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_glo(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_in,*cloud_imu,right_vlp_2_imu);
    
    //TODO 找到当前时刻点云对应的位置
    double stamp = msg_in->header.stamp.toSec();
    Eigen::Matrix4f current_pose;
    if (!get_pose_from_stamp(stamp,current_pose)) {
      LOG(WARNING) << "this cloud msg pose not find.skip.";
      return;
    }
    pcl::transformPointCloud(*cloud_imu,*cloud_glo,current_pose);
    LOG(INFO) << "current pose : \n" << current_pose;
    //pub cloud imu 
    sensor_msgs::PointCloud2 cloud_imu_msg;
    pcl::toROSMsg(*cloud_imu,cloud_imu_msg);
    cloud_imu_msg.header.stamp = msg_in->header.stamp;
    cloud_imu_msg.header.frame_id = "imu";
    pub_right_velo.publish(cloud_imu_msg);

    //pub cloud global 
    sensor_msgs::PointCloud2 cloud_glo_msg;
    pcl::toROSMsg(*cloud_glo,cloud_glo_msg);
    cloud_glo_msg.header.stamp = msg_in->header.stamp;
    cloud_glo_msg.header.frame_id = "global";
    pub_right_velo_glo.publish(cloud_glo_msg);
  }
  bool get_pose_from_stamp(const double _stamp,Eigen::Matrix4f &_current_pose) {
    auto map_it = stamp_pose_map.begin();
    auto former_it = stamp_pose_map.begin();
    int count = 0;
    while(map_it != stamp_pose_map.end()) {
      if(map_it->first < _stamp) {
        map_it ++;
        if (count > 0) {
          former_it++;
        }
        count ++;
      }
      else break;
    }
    double former_stamp = former_it->first;
    double former_gap = _stamp - former_stamp;
    double next_gap = map_it->first - _stamp;
    if (former_gap > 0.1 || next_gap > 0.1 ) {
      return false;
    }
    LOG(INFO) << "former_gap = " << former_gap;
    LOG(INFO) << "next_gap = " << next_gap;
    double front_ratio = (next_gap) / (next_gap + former_gap);
    Eigen::Matrix4f res_pose = Eigen::Matrix4f::Identity();
    if (!interpolate_pose(former_it->second.cast<float>(),map_it->second.cast<float>(),front_ratio,res_pose)) {
      return false;
    }
    if (map_it == stamp_pose_map.end()) {
      return false;
    }
    res_pose.block<3,1>(0,3) -= stamp_pose_map.begin()->second.block<3,1>(0,3).cast<float>();
    _current_pose = res_pose;
    return true;
  }
  bool interpolate_pose(const Eigen::Matrix4f _front_pose,const Eigen::Matrix4f _back_pose,const double _front_ratio,Eigen::Matrix4f& _interpolated_pose) {
    if(_front_ratio < 0 || _front_ratio > 1) {
      return false;
    }
    double back_ratio_ = 1 - _front_ratio;
    Eigen::Vector3f interpolate_t_ = back_ratio_ * _back_pose.block<3,1>(0,3) + _front_ratio * _front_pose.block<3,1>(0,3);
    Eigen::Quaternionf front_q_ = Eigen::Quaternionf(_front_pose.block<3,3>(0,0));
    Eigen::Quaternionf back_q_ = Eigen::Quaternionf(_back_pose.block<3,3>(0,0));
    Eigen::Quaternionf interpolate_q_ = front_q_.slerp(_front_ratio,back_q_);
  
    _interpolated_pose = gen_affile_matrix_from_q_t(interpolate_q_,interpolate_t_);
  
    return true;
  }
  static Eigen::Matrix4f gen_affile_matrix_from_q_t(const Eigen::Quaternionf& _rot,const Eigen::Vector3f _t) {
  Eigen::Matrix4f out_ = Eigen::Matrix4f::Identity();
  out_.block<3,3>(0,0) = Eigen::Matrix3f(_rot);
  out_.block<3,1>(0,3) = Eigen::Vector3f(_t);
  return out_;
  }
 private:
  ros::NodeHandle nh;
  ros::Subscriber sub_vlp_right;
  ros::Subscriber sub_vlp_left;
  ros::Publisher pub_left_velo;
  ros::Publisher pub_right_velo;
  ros::Publisher pub_left_velo_glo;
  ros::Publisher pub_right_velo_glo;

  map<double,Matrix4d> stamp_pose_map;
  Eigen::Matrix4d Vehicle2LeftVLP;
  Eigen::Matrix4d Vehicle2RightVLP;
  Eigen::Matrix4d Vehicle2IMU;
};

#endif