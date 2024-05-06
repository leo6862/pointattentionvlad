#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <glog/logging.h>

#include <kaist_yyj/kaist_vlp_receiver.h>

int main(int argc,char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  ros::init(argc,argv,"main_node");
  ros::NodeHandle nh;

  std::string global_file_path = "/home/yyj/download/kaist/urban17/urban17_pose/urban17/global_pose.csv";
  Kaist_vlp_reciever vlp_reciever(nh,global_file_path);

  ros::spin();
}

