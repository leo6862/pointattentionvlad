/*
  input cloud : n *3 , target_point_num : m
  vector<int> sampled_cloud; //用于存储 采样后的点云在原始点云中的index
  vector<double> distance(n,max);
  生成一个当前 点云 index 的一个随机数
  
  for(int ind = 0;ind < m;ind++) {
    1. 获取当前采样点的坐标 point
    2. 计算点云中所有点 与当前 point 的距离
    3. 
  }
*/


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/distances.h>

#include <vector>
#include <random>

using namespace std;
using Point=pcl::PointXYZ;
using PointCloud=pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr=pcl::PointCloud<pcl::PointXYZ>::Ptr;

// def farthest_point_sample(xyz, npoint):
//     """
//     Input:
//         xyz: pointcloud data, [B, N, 3]
//         npoint: number of samples
//     Return:
//         centroids : sampled pointcloud index, [B, npoint]
//     """
//     device = xyz.device
//     B, N, C = xyz.shape
//     centroids = torch.zeros(B, npoint,dtype=torch.long).to(device)
//     distance = torch.ones(B, N).to(device) * 1e10
//     #生成了一个大小为B的长整型张量（tensor），其中每个元素的取值范围是[0, N)，即0到N-1之间的整数
//     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
//     batch_indices = torch.arange(B, dtype=torch.long).to(device)
//     for i in range(npoint):
//         #将res中的第i个点设置为 farthest
//         centroids[:, i] = farthest
//         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
//         dist = torch.sum((xyz - centroid) ** 2, -1)
//         distance = torch.min(distance, dist) #yyj 计算两个tensor 逐元素的最小值
//         farthest = torch.max(distance, -1)[1] #获取 distance 中的最大元素用于
//     return centroids
bool fps(PointCloudPtr _cloud_in,int _target_num,vector<int>& _sample_index) {
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
    for(int p_ind = 0;p_ind < _cloud_in->size();p_ind++) {
      dist[p_ind] = pcl::euclideanDistance(farthest_point,_cloud_in->points[p_ind]);
    }
    
    for(int dist_ind = 0;dist_ind < distance.size();dist_ind++) {
      distance[dist_ind] = std::min<double>(dist[dist_ind],distance[dist_ind]);
    }
    auto max_dist = std::max_element(distance.begin(), distance.end());

    // 获取最大值元素的下标
    farthest_ind = std::distance(distance.begin(), max_dist);
  }
  return true;
}


int main(int argc,char** argv) {
  /*
   
  */
  PointCloudPtr cloud(new PointCloud());
  cloud->push_back(Point(0,0,0));
  cloud->push_back(Point(0,5,0));
  cloud->push_back(Point(5,0,0));
  cloud->push_back(Point(5,5,0));

  cloud->push_back(Point(1,1,0));
  cloud->push_back(Point(4,1,0));
  cloud->push_back(Point(1,4,0));
  cloud->push_back(Point(4,4,0));

  cloud->push_back(Point(2,2,0));
  cloud->push_back(Point(3,2,0));
  cloud->push_back(Point(2,3,0));
  cloud->push_back(Point(3,3,0));
  
  int target_num = 6;

  vector<int> sample_index;
  fps(cloud,target_num,sample_index);

  PointCloudPtr sampled_cloud(new PointCloud);
  for(int p_ind = 0;p_ind < sample_index.size();p_ind ++ ) {
    sampled_cloud->push_back(cloud->points[sample_index[p_ind]]);
    std::cout << "sample point : " << cloud->points[sample_index[p_ind]].x 
                            << " " << cloud->points[sample_index[p_ind]].y 
                            << " " << cloud->points[sample_index[p_ind]].z;
  }
  

}


