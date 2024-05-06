#ifndef KAIST_YYJ_UTILS_H
#define KAIST_YYJ_UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <chrono>
#include <sstream>
#include <string>
#include <iostream>
#include <random>
#include <glog/logging.h>

#include <omp.h>

#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/PointIndices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/distances.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h> // 拟合平面

using namespace std;
using Point=pcl::PointXYZ;
using PointCloud=pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr=pcl::PointCloud<pcl::PointXYZ>::Ptr;

namespace utils{

struct Kaist_point {
  float x;
  float y;
  float z;
  float intensity;
};

static bool load_kaist_cloud(const std::string filename,PointCloudPtr& _cloud) {
  std::ifstream file(filename, std::ios::binary);
  if (!file)
  {
    std::cerr << "Failed to open the file." << std::endl;
    return false;
  }
  // Read the binary data into a vector of PointXYZIT
  std::vector<Kaist_point> pointCloud;
  Kaist_point point;
  while (file.read((char*)&point, sizeof(Kaist_point)))
  {
    pointCloud.push_back(point);
  }
  // Close the file
  file.close();
  PointCloudPtr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // Access and process the loaded point cloud data
  for (const auto& point : pointCloud)
  {
    cloud->push_back(pcl::PointXYZ(point.x,point.y,point.z));
  }
  *_cloud = *cloud;
  return true;
}

static Eigen::Matrix4f gen_affile_matrix_from_q_t(const Eigen::Quaternionf& _rot,const Eigen::Vector3f _t) {
  Eigen::Matrix4f out_ = Eigen::Matrix4f::Identity();
  out_.block<3,3>(0,0) = Eigen::Matrix3f(_rot);
  out_.block<3,1>(0,3) = Eigen::Vector3f(_t);
  return out_;
}

__attribute__((optimize("O0"))) static bool ground_filter(PointCloudPtr _cloud_in,PointCloudPtr _non_ground_cloud) {
  /*
    提取当前 cloud中高度最小的 40%的点云 ， 使用ransac拟合平面。
    将原始点云中符合当前平面方程的点擦除
  */
  
  //提取当前点云中高度最低的 40% 的点
  PointCloudPtr sample_cloud(new PointCloud);
  //省略赋值操作 按x坐标值从小到大排序
  std::sort(_cloud_in->begin(), _cloud_in->end(),
  [](Point pt1,Point pt2) {return pt1.z < pt2.z; });
  
  int sample_num = 0.4 * _cloud_in->size();
  for(int p_ind = 0;p_ind < sample_num;p_ind++) {
    sample_cloud->push_back(_cloud_in->points[p_ind]);
  }

  // pcl::SampleConsensusModelPlane<Point>::Ptr model_plane(new pcl::SampleConsensusModelPlane<Point>(sample_cloud));
	// pcl::RandomSampleConsensus<Point> ransac(model_plane);	
	// ransac.setDistanceThreshold(0.3);	//设置距离阈值，与平面距离小于0.1的点作为内点
	// ransac.computeModel();				//执行模型估计
	// //-------------------------根据索引提取内点--------------------------
	// PointCloudPtr cloud_plane(new PointCloud);
	// vector<int> inliers;				//存储内点索引的容器
	// ransac.getInliers(inliers);			//提取内点索引
  // pcl::ExtractIndices<pcl::PointXYZ> extract;
  // extract.setInputCloud (_cloud_in);
  // pcl::PointIndices::Ptr inlier_ (new pcl::PointIndices ());
  // inlier_->indices = inliers;
  // extract.setIndices (inlier_);
  // extract.setNegative (true);
  // extract.filter (*_non_ground_cloud);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr indices(new pcl::PointIndices());
	pcl::SACSegmentation<pcl::PointXYZ> sacseg;
	sacseg.setOptimizeCoefficients(true);
	sacseg.setModelType(pcl::SACMODEL_PLANE);
	sacseg.setMethodType(pcl::SAC_RANSAC);
	sacseg.setMaxIterations(100);
	sacseg.setDistanceThreshold(0.1);
	sacseg.setInputCloud(sample_cloud);
	sacseg.segment(*indices,*coefficients);
  // _non_ground_cloud.reset(new PointCloud());
  double sqrt_coes = sqrt(coefficients->values[0] * coefficients->values[0] + coefficients->values[1] * coefficients->values[1]
                        + coefficients->values[2] * coefficients->values[2]);
  for(const auto& point : _cloud_in->points) {
    double dist_ = abs(point.x * coefficients->values[0] + point.y * coefficients->values[1] + point.z * coefficients->values[2] +  coefficients->values[3]) / sqrt_coes;
    // LOG(INFO) << "dist_ = " << dist_;
    if (dist_ > 0.6) {
      _non_ground_cloud->push_back(point);
    }
  }
  
}

static bool interpolate_pose(const Eigen::Matrix4f _front_pose,const Eigen::Matrix4f _back_pose,const double _front_ratio,Eigen::Matrix4f& _interpolated_pose) {
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

//only used in kaist dataset
static bool load_calib_from_file(const std::string _calib_file_path , Eigen::Matrix4f& _t) {
  std::vector<double> T_vector;
  std::vector<double> R_vector;
  std::ifstream file(_calib_file_path);
  if (!file) {
    return false;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.find("T:") != std::string::npos) {
        std::istringstream iss(line);
        std::string header;
        double value;
        while (iss >> header) {
          if(header != "T:")
            T_vector.push_back(std::stod(header));
        }
    }
    if (line.find("R:") != std::string::npos) {
      std::istringstream iss(line);
      std::string header;
      double value;
      while (iss >> header) {
        if(header != "R:")
          R_vector.push_back(std::stod(header));
      }
    }
  }
  file.close();
  Eigen::Matrix3f rot;
  rot << R_vector[0] ,R_vector[1] ,R_vector[2] ,
         R_vector[3] ,R_vector[4] ,R_vector[5] ,
         R_vector[6] ,R_vector[7] ,R_vector[8] ;
  Eigen::Vector3f t;
  t << T_vector[0] , T_vector[1], T_vector[2];
  _t.block<3,3>(0,0) = rot;
  _t.block<3,1>(0,3) = t;
  return true;
  }
static bool fps(PointCloudPtr _cloud_in,int _target_num,vector<int>& _sample_index) {
  int point_num = _cloud_in->size();
  _sample_index.clear();
  if (_target_num > point_num) {
    return false;
  }
  std::vector<double> distance(point_num,9999999999999999999);
  std::random_device rd;  
  std::mt19937 gen(rd()); 
  std::uniform_int_distribution<int> distribution(0, point_num - 1);
  int farthest_ind = distribution(gen);  // 生成随机整数 ， 作为第一个采样点

  for(int sample_ind = 0;sample_ind < _target_num;sample_ind++) {
    _sample_index.push_back(farthest_ind);
    Point farthest_point = _cloud_in->points[farthest_ind];

    //计算点云中所有点与该点的距离

    vector<double> dist(_cloud_in->size());
    //!
    // #ifdef MP_EN
    //     omp_set_num_threads(MP_PROC_NUM);
    //     #pragma omp parallel for
    // #endif
    for(int p_ind = 0;p_ind < _cloud_in->size();p_ind++) {
      dist[p_ind] = pcl::euclideanDistance(farthest_point,_cloud_in->points[p_ind]);
    }
    // #ifdef MP_EN
    //     omp_set_num_threads(MP_PROC_NUM);
    //     #pragma omp parallel for
    // #endif
    for(int dist_ind = 0;dist_ind < distance.size();dist_ind++) {
      distance[dist_ind] = std::min<double>(dist[dist_ind],distance[dist_ind]);
    }
    auto max_dist = std::max_element(distance.begin(), distance.end());

    // 获取最大值元素的下标
    farthest_ind = std::distance(distance.begin(), max_dist);
  }
  return true;
}

class TicToc{
public:
TicToc(){
t_start_ = std::chrono::steady_clock::now();
}

double toc(){
t_end_ = std::chrono::steady_clock::now();
time_used_ = std::chrono::duration<double>(t_end_ - t_start_).count();
t_start_ = std::chrono::steady_clock::now();
return time_used_;
}

private:
std::chrono::steady_clock::time_point t_start_ ;
std::chrono::steady_clock::time_point t_end_;
double time_used_;
};

}
#endif