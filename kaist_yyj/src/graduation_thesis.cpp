
#include <kaist_yyj/utils.h>
#include <kaist_yyj/submap_generator.h>

#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace Eigen;

#include <kaist_yyj/submap_generator.h>

int main(int argc,char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir="/home/yyj/catkin_ws/src/kaist_yyj/log";
  string yaml_file = "/home/yyj/catkin_ws/src/kaist_yyj/config/submap_generator.yaml";
  YAML::Node node = YAML::LoadFile(yaml_file);
  std::vector<std::string> seqs = node["seqs"].as<vector<string>>();
  

  //TODO  load seqs from yaml
  for(int seq_ind = 0;seq_ind < seqs.size();seq_ind++) {
    LOG(INFO) << "processing : " << seqs[seq_ind];
    Submap_generator map_generator(node,seqs[seq_ind]);
    map_generator.graduation_thesis_vis();
    LOG(INFO) << seqs[seq_ind] << " done.";
  }
}
