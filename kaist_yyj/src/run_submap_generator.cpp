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
    if(map_generator.run()) {
      LOG(INFO) << seqs[seq_ind] << " done.";
    }
    else{
      LOG(ERROR) << seqs[seq_ind] << " failed.";
    }
  }
}

