#include <kaist_yyj/map_generator.h>

int main(int argc,char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  ros::init(argc,argv,"map_generator_node");
  ros::NodeHandle nh;
  string yaml_file = "/home/yyj/catkin_ws/src/kaist_yyj/config/config.yaml";
  YAML::Node node = YAML::LoadFile(yaml_file);
  std::string global_file_path = "/home/yyj/download/kaist/urban17/urban17_pose/urban17/global_pose.csv";
  Map_generator map_generator(nh,node);

  ros::spin();
}

