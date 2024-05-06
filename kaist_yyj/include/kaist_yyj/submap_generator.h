#ifndef KAIST_YYJ_MAP_GENERATOR_H
#define KAIST_YYJ_MAP_GENERATOR_H

#include <kaist_yyj/utils.h>
#include <kaist_yyj/ground_filter.h>

#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <iomanip>
#include <vector>
#include <future>
#include <algorithm>
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

struct Cloud_file {
  
  Cloud_file (const std::string _file_path,const bool _is_ns1,const double _stamp) {
    file_path = _file_path;
    is_ns1 = _is_ns1;
    stamp = _stamp;
  }
  std::string file_path; //abs path 
  bool is_ns1; // ns1 : right_vlp , ns2 : left_vlp
  double stamp;
};

static bool compare_cloud_file(const Cloud_file& data1, const Cloud_file& data2)
{
    return data1.stamp < data2.stamp;
}

struct Cloud_data {
  Cloud_data(PointCloudPtr _cloud_ptr,Pose_stamped _ps,bool _is_ns1) {
    cloud_ptr = _cloud_ptr;
    pose_stamp = _ps;
    is_ns1 = _is_ns1;
  }
  Cloud_data(PointCloudPtr _cloud_ptr,const double _stamp,const Eigen::Matrix4f _pose,bool _is_ns1) {
    cloud_ptr = _cloud_ptr;
    pose_stamp = Pose_stamped(_stamp,_pose);
    is_ns1 = _is_ns1;
  }
  bool is_ns1;
  PointCloudPtr cloud_ptr;
  Pose_stamped pose_stamp;
};

class Submap_generator{
 public:
  Submap_generator(const YAML::Node& _node,const std::string seq);
  bool run();
  void graduation_thesis_vis();
 private:
  bool load_global_pose(const string _global_pose_path);
  bool get_pose_from_stamp(const double _stamp,int& _ind_in_q,Eigen::Matrix4f& _p);
  bool check_if_key_frame(const Matrix4f _prev_key_frame_pose,const Matrix4f _current_pose);
  bool gen_submap(PointCloudPtr _submap);
  bool post_process(PointCloudPtr _submap_cloud);
  bool save_submap(PointCloudPtr _submap_cloud);
  bool save_submap(Cloud_data& _cloud_data);
  bool fps(PointCloudPtr _cloud_in,int _target_num,vector<int>& _sample_index);
  string extractFilename(const std::string& path);
  Cloud_data get_cloud_data_from_file(const Cloud_file _cloud_file,const Matrix4f current_cloud_pose);
  void remove_old_cloud_data();
 private:
  YAML::Node node;

  deque<Pose_stamped> stamp_gt_poses;
  deque<Cloud_data> cloud_data_q;
  std::vector<Cloud_data> submap_collect;

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