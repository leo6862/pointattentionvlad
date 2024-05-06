#ifndef OCTOMAP_DRAFT_GROUND_FILTER_H
#define OCTOMAP_DRAFT_GROUND_FILTER_H

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>

#include <Eigen/Dense>


#include <glog/logging.h>
#include <yaml-cpp/yaml.h>


namespace ground_filter{
struct PointXYZLabel
{
  PCL_ADD_POINT4D;                // quad-word XYZ
  uint16_t label;                 ///< point label
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
} EIGEN_ALIGN16;
}
POINT_CLOUD_REGISTER_POINT_STRUCT(ground_filter::PointXYZLabel,
                                  (float, x, x)(float, y, y)(float, z, z)(uint16_t, label, label))

#define  PointXYZLabel ground_filter::PointXYZLabel 
namespace utils{
template<typename PointT>
class Ground_filter{
public:
  Ground_filter(YAML::Node& node) {
    clip_height = node["clip_height"].as<double>();
    sensor_height = node["sensor_height"].as<double>();
    min_distance = node["min_distance"].as<double>();
    max_distance = node["max_distance"].as<double>();
    num_iter = node["num_iter"].as<int>();
    num_lpr = node["num_lpr"].as<int>();
    th_seeds = node["th_seeds"].as<double>();
    th_dist = node["th_dist"].as<double>();

    ground_cloud.reset(new pcl::PointCloud<PointT>);
    non_ground_cloud.reset(new pcl::PointCloud<PointT>);
  }

  __attribute__((optimize("O0"))) void ground_filtering(typename pcl::PointCloud<PointT>::Ptr& cloud_in,typename pcl::PointCloud<PointT>::Ptr& ground_cloud_,typename pcl::PointCloud<PointT>::Ptr& non_ground_cloud_) {
    cloud_in_backup.reset(new pcl::PointCloud<PointT>(*cloud_in));
    PointXYZLabel point_label;
    cloud_in_label.reset(new pcl::PointCloud<PointXYZLabel>);
    for(int i = 0;i < cloud_in->size();i++){
      point_label.x = cloud_in->points[i].x;
      point_label.y = cloud_in->points[i].y;
      point_label.z = cloud_in->points[i].z;
      point_label.label = 0u;
      cloud_in_label->push_back(point_label);
    }
    sort(cloud_in->points.begin(), cloud_in->end(), point_cmp);
    typename pcl::PointCloud<PointT>::iterator it = cloud_in->points.begin();
    for (int i = 0; i < cloud_in->points.size(); i++)
    {
        if (cloud_in->points[i].z < -1.5 * sensor_height)
        {
            it++;
        }
        else
        {
            break;
        }
    }
    cloud_in->points.erase(cloud_in->points.begin(), it);

    extract_intial_seeds(cloud_in);
    ground_cloud = seeds_cloud;

     for (int i = 0; i < num_iter; i++)
    {
        estimate_plane();
        ground_cloud->clear();
        non_ground_cloud->clear();

        //pointcloud to matrix
        Eigen::MatrixXf points(cloud_in_backup->points.size(), 3);
        int j = 0;
        for (auto p : cloud_in_backup->points)
        {
            points.row(j++) << p.x, p.y, p.z;
        }
        // ground plane model
        Eigen::VectorXf result = points * normal;
        // threshold filter
        for (int r = 0; r < result.rows(); r++)
        {
            if (result[r] < th_dist_d)
            {
                cloud_in_label->points[r].label = 1u; // means ground
                ground_cloud->points.push_back(cloud_in_backup->points[r]);
            }
            else
            {
                cloud_in_label->points[r].label = 0u; // means not ground and non clusterred
                non_ground_cloud->push_back(cloud_in_backup->points[r]);
            }
        }
    }

    final_non_ground_cloud.reset(new pcl::PointCloud<PointT>);
    post_process(non_ground_cloud,final_non_ground_cloud);
    

//    non_ground_cloud_->width = 1;
//    non_ground_cloud_->height = non_ground_cloud_->size();
    non_ground_cloud_ = final_non_ground_cloud;
    ground_cloud_ = ground_cloud; 
  }

private:
  __attribute__((optimize("O0"))) void extract_intial_seeds(typename pcl::PointCloud<PointT>::Ptr& cloud) {
    seeds_cloud.reset(new pcl::PointCloud<PointT>);

    double sum = 0;
    int cnt = 0;
    // Calculate the mean height value.
    for (int i = 0; i < cloud->points.size() && cnt < num_lpr; i++)//num_lpr_默认为20
    {
        sum += cloud->points[i].z;
        cnt++;
    }
    double lpr_height = cnt != 0 ? sum / cnt : 0; // in case divide by 0
    seeds_cloud->clear();
    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (int i = 0; i < cloud->points.size(); i++)
    {
        if (cloud->points[i].z < lpr_height + th_seeds) //th_seeds默认为1.2
        {
            seeds_cloud->points.push_back(cloud->points[i]);
        }
    }
  }
  __attribute__((optimize("O0"))) void estimate_plane() {
    Eigen::Matrix3f cov;
    Eigen::Vector4f pc_mean;
    pcl::computeMeanAndCovarianceMatrix(*ground_cloud, cov, pc_mean);
    // Singular Value Decomposition: SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    normal = (svd.matrixU().col(2));
    // mean ground seeds value
    Eigen::Vector3f seeds_mean = pc_mean.head<3>();

    // according to normal.T*[x,y,z] = -d
    d = -(normal.transpose() * seeds_mean)(0, 0);
    // set distance threhold to `th_dist - d`
    th_dist_d = th_dist - d;
  }
  __attribute__((optimize("O0"))) void post_process(const typename pcl::PointCloud<PointT>::Ptr in, const typename pcl::PointCloud<PointT>::Ptr out) {
    typename pcl::PointCloud<PointT>::Ptr cliped_pc_ptr(new pcl::PointCloud<PointT>);
    clip_above(in, cliped_pc_ptr);
    typename pcl::PointCloud<PointT>::Ptr remove_close(new pcl::PointCloud<PointT>);
    remove_close_far_pt(cliped_pc_ptr, out);
  } 
  __attribute__((optimize("O0"))) void remove_close_far_pt(const typename pcl::PointCloud<PointT>::Ptr in,
                                            const typename pcl::PointCloud<PointT>::Ptr out) {
    pcl::ExtractIndices<PointT> cliper;

    cliper.setInputCloud(in);
    pcl::PointIndices indices;
    #pragma omp for
    for (size_t i = 0; i < in->points.size(); i++)
    {
        double distance = sqrt(in->points[i].x * in->points[i].x + in->points[i].y * in->points[i].y);

        if ((distance < min_distance) || (distance > max_distance))
        {
            indices.indices.push_back(i);
        }
    }
    cliper.setIndices(boost::make_shared<pcl::PointIndices>(indices));
    cliper.setNegative(true); //ture to remove the indices
    cliper.filter(*out);
  }
  __attribute__((optimize("O0"))) void clip_above(const typename pcl::PointCloud<PointT>::Ptr in,const typename pcl::PointCloud<PointT>::Ptr out) {
    pcl::ExtractIndices<PointT> cliper;

    cliper.setInputCloud(in);
    pcl::PointIndices indices;
    #pragma omp for
    for (size_t i = 0; i < in->points.size(); i++)
    {
        if (in->points[i].z > clip_height)
        {
            indices.indices.push_back(i);
        }
    }
    cliper.setIndices(boost::make_shared<pcl::PointIndices>(indices));
    cliper.setNegative(true); //ture to remove the indices
    cliper.filter(*out);
  }
  __attribute__((optimize("O0"))) static bool point_cmp(PointT a, PointT b)
    {
        return a.z < b.z;
    }   

private:
  double clip_height;
  double sensor_height;
  double min_distance;
  double max_distance;
  int num_iter;
  int num_lpr;
  double th_seeds;
  double th_dist; 

  float d, th_dist_d;
  typename pcl::PointCloud<PointT>::Ptr cloud_in_backup;
  pcl::PointCloud<PointXYZLabel>::Ptr cloud_in_label;
  typename pcl::PointCloud<PointT>::Ptr seeds_cloud;
  typename pcl::PointCloud<PointT>::Ptr ground_cloud;
  typename pcl::PointCloud<PointT>::Ptr non_ground_cloud;

  typename pcl::PointCloud<PointT>::Ptr final_non_ground_cloud;
  Eigen::MatrixXf normal;
};

}
#endif