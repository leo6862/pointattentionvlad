#ifndef KAIST_YYJ_MAP_GENERATOR_H
#define KAIST_YYJ_MAP_GENERATOR_H

#include <kaist_yyj/utils.h>
#include <kaist_yyj/ground_filter.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <iomanip>
#include <vector>
#include <filesystem>
#include <limits>
#include <queue>
#include <deque>
#include <glog/logging.h>

using namespace std;
using namespace Eigen;

using Point = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<Point>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;

struct Pose_stamped {
  Pose_stamped() {}
  Pose_stamped(const double _s,const Matrix4f _p) {
    stamp = _s;
    pose = _p;
  }
  double stamp;
  Eigen::Matrix4f pose;
};

struct Cloud_data {
  Cloud_data(PointCloudPtr _cloud_ptr,Pose_stamped _ps) {
    cloud_ptr = _cloud_ptr;
    pose_stamp = _ps;
  }
  Cloud_data(PointCloudPtr _cloud_ptr,const double _stamp,const Eigen::Matrix4f _pose) {
    cloud_ptr = _cloud_ptr;
    pose_stamp = Pose_stamped(_stamp,_pose);
  }
  PointCloudPtr cloud_ptr;
  Pose_stamped pose_stamp;
};

class Map_generator{
 public:
  Map_generator(const ros::NodeHandle& _nh,const YAML::Node& _node);
 private:
  void ns1_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& _msg_in);
  void ns2_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& _msg_in);
  bool load_global_pose(const string _global_pose_path);
  bool get_pose_from_stamp(const double _stamp,int& _ind_in_q,Eigen::Matrix4f& _p);
  bool check_if_key_frame(const Matrix4f _prev_key_frame_pose,const Matrix4f _current_pose);
  bool gen_submap(PointCloudPtr _submap);
  bool post_process(PointCloudPtr _submap_cloud);
  bool save_submap(PointCloudPtr _submap_cloud);
 private:
  ros::NodeHandle nh;
  YAML::Node node;
  ros::Subscriber sub_cloud_ns1;
  ros::Subscriber sub_cloud_ns2;
  ros::Publisher pub_submap_glo;

  deque<Pose_stamped> stamp_gt_poses;
  deque<Cloud_data> cloud_data_q;
  
  string seq_data_root; ///home/yyj/download/kaist/
  string current_seq_data_root; ///home/yyj/download/kaist/urban17/
  string current_seq; //"urban17"
  string cloud_save_dir; 
  string txt_save_dir;

  Matrix4f prev_ns1_kf_pose = Eigen::Matrix4f::Identity(); //only add new frame when new frame pose is signficant different from current pose
  Matrix4f prev_ns2_kf_pose = Eigen::Matrix4f::Identity();
  Matrix4f T_ns1_imu = Eigen::Matrix4f::Identity();
  Matrix4f T_ns2_imu = Eigen::Matrix4f::Identity();
  Matrix4f T_ego_ns1 = Eigen::Matrix4f::Identity();
  Matrix4f T_ego_ns2 = Eigen::Matrix4f::Identity();
  Matrix4f T_ego_imu = Eigen::Matrix4f::Identity();
  Vector3f last_submap_pose = Vector3f::Zero();
  Vector3f gt_pose_offset; // 为了可视化 方便 ， 对于当前的每个gt pose都进行了偏移，但最后生成数据集的时候需要将偏移量加回来
  
  int submap_frame_num;
  int submap_point_num;
  double submap_cube_size;
  double last_submap_stamp;
  double submap_gap;
  std::shared_ptr<utils::Ground_filter<Point>> ground_filter_ptr; //not used 

  ofstream of_stamp_pose;
};

#endif 