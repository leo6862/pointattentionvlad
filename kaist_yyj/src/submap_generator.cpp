#include <kaist_yyj/submap_generator.h>

Submap_generator::Submap_generator(const YAML::Node& _node ,const std::string seq) {
  /*
    1. load global pose 
    2. 
  */
  node = _node;  
  current_seq           = seq;   //"urban17"
  seq_data_root         = _node["seq_data_root"].as<string>(); ///home/yyj/download/kaist/
  current_seq_data_root = seq_data_root + current_seq + "/"; // /home/yyj/download/kaist/urban17/
  txt_save_dir = string(PROJECT_PATH) + "/data/";
  cloud_save_dir = string(PROJECT_PATH) + "/data/" + current_seq;
  if (!std::filesystem::exists(std::filesystem::path(txt_save_dir))) {
    std::filesystem::create_directory(std::filesystem::path(txt_save_dir));
  } 
  if (!std::filesystem::exists(std::filesystem::path(cloud_save_dir))) {
    std::filesystem::create_directory(std::filesystem::path(cloud_save_dir));
  } 
  of_stamp_pose = std::ofstream(txt_save_dir + current_seq + ".txt");
  if(!of_stamp_pose) {
    LOG(ERROR) << "open output file failed . exit.";
    exit(0);
  }

  submap_frame_num = _node["submap_frame_num"].as<int>();
  submap_cube_size = _node["submap_cube_size"].as<double>();
  submap_gap       = _node["submap_gap"].as<double>();
  submap_point_num = _node["submap_point_num"].as<int>();
  
  //Load global pose
  if (!load_global_pose(current_seq_data_root + "global_pose.csv")) {
    LOG(ERROR) << "load global pose txt failed. make sure you set the correct global_patth file path.";
    LOG(ERROR) << "current pose file path : " << current_seq_data_root + "global_pose.csv";
    exit(0);
  }

  //Load calibration file 
  string ego_ns1_file_path = current_seq_data_root + "calibration/Vehicle2RightVLP.txt";
  string ego_ns2_file_path = current_seq_data_root + "calibration/Vehicle2LeftVLP.txt";
  string ego_imu_file_path = current_seq_data_root + "calibration/Vehicle2IMU.txt";
  if (!utils::load_calib_from_file(ego_ns1_file_path,T_ego_ns1)) {
    LOG(ERROR) << "Load Vehicle2Right txt failed.";
    exit(0);
  }
  if (!utils::load_calib_from_file(ego_ns2_file_path,T_ego_ns2)) {
    LOG(ERROR) << "Load Vehicle2Left txt failed.";
    exit(0);
  }
  if (!utils::load_calib_from_file(ego_imu_file_path,T_ego_imu)) {
    LOG(ERROR) << "Load Vehicle2IMU txt failed.";
    exit(0);
  }
  // LOG(INFO) << "T_ego_imu : \n" << T_ego_imu;
  // LOG(INFO) << "T_ego_ns1 : \n" << T_ego_ns1;
  // LOG(INFO) << "T_ego_ns2 : \n" << T_ego_ns2;
  T_ns1_imu = T_ego_imu.inverse() * T_ego_ns1;
  T_ns2_imu = T_ego_imu.inverse() * T_ego_ns2;

  ground_filter_ptr = std::make_shared<utils::Ground_filter<Point>>(node);
}

// entry
__attribute__((optimize("O0"))) bool Submap_generator::run() {
  /*
    1. 根据当前seq的地址 ， 读取所有的 right_vlp , left vlp 中的 bin文件 
      stamp -> (ns1 or ns2 , abs_path)
    2. 将所有的点云的 加入到一个队列中 ， 依次弹出 ，
        2.1 trans to imu frame 
        2.2 query pose -> trans to global frame
        2.3 if current pose key frame （ns1 or ns2） add into cloud_data_q
        2.4 if current frame pose differ from last_submap_pose 
            generate new submap 
            save submap 
  */

  //1. load point cloud bin file , generate vector & sort it by stamp 
  vector<Cloud_file> cloud_files;
  std::string ns1_cloud_dir = current_seq_data_root + "sensor_data/VLP_right/";
  std::string ns2_cloud_dir = current_seq_data_root + "sensor_data/VLP_left/";

  //检查 两个文件夹是否存在
  if (!std::filesystem::exists(std::filesystem::path(ns1_cloud_dir))) {
    LOG(ERROR) << ns1_cloud_dir << " not exist. skip.";
    return false;
  }
  if (!std::filesystem::exists(std::filesystem::path(ns2_cloud_dir))) {
    LOG(ERROR) << ns2_cloud_dir << " not exist. skip.";
    return false;
  }

  //将两个文件夹中的文件全部 加入到 cloud_files中
  for (const auto& entry : std::filesystem::directory_iterator(ns1_cloud_dir)){
    string abs_file_path_ = entry.path();
    double stamp_ = stod(extractFilename(abs_file_path_)) * 1e-9;
    cloud_files.emplace_back(abs_file_path_,true,stamp_);
  }
  for (const auto& entry : std::filesystem::directory_iterator(ns2_cloud_dir)){
    string abs_file_path_ = entry.path();
    double stamp_ = stod(extractFilename(abs_file_path_)) * 1e-9;
    cloud_files.emplace_back(abs_file_path_,false,stamp_); 
  }
  std::sort(cloud_files.begin(), cloud_files.end(), compare_cloud_file); //根据时间戳 对 cloud_files进行排序
  
  // 2 for loop for every cloud
  for(int cloud_file_ind = 0;cloud_file_ind < cloud_files.size();cloud_file_ind ++) {
    LOG(INFO) << current_seq << "  processing " << cloud_file_ind << "  /  " << cloud_files.size();
    Cloud_file current_cloud_file = cloud_files[cloud_file_ind];
    Matrix4f current_cloud_pose = Matrix4f::Identity();
    
    int stamp_ind;  //not used 
    if(!get_pose_from_stamp(current_cloud_file.stamp,stamp_ind,current_cloud_pose)) {
      continue;
    }
    // LOG(INFO) << "current_cloud_pose = \n" << current_cloud_pose;
    if (current_cloud_file.is_ns1) {
      /*
        当前文件 ns1 
      */
      if (check_if_key_frame(prev_ns1_kf_pose,current_cloud_pose)) {
        prev_ns1_kf_pose = current_cloud_pose;
      }
      else {
        continue;
      }
    }
    else {
      /*
        ns2  
      */
      //当前帧是关键帧 进行处理
      if (check_if_key_frame(prev_ns2_kf_pose,current_cloud_pose)) {
        prev_ns2_kf_pose = current_cloud_pose;
      }
      else {
        //如果当前帧不是关键帧 就不进行处理了
        continue;
      }
    }
    
    // add into cloud_data_q 
    cloud_data_q.push_front(get_cloud_data_from_file(current_cloud_file,current_cloud_pose));
    // LOG(INFO) << "current cloud size = " << cloud_data_q[0].cloud_ptr->size();
    PointCloudPtr submap_cloud_(new PointCloud);
    if(!gen_submap(submap_cloud_)) {
      continue;
    }
    // LOG(INFO) << "submap_cloud_ size : " << submap_cloud_->size();
    Eigen::Matrix4f submap_pose_ = Matrix4f::Identity();
    submap_pose_.block<3,1>(0,3) = last_submap_pose.cast<float>();
    submap_collect.emplace_back(submap_cloud_,last_submap_stamp,submap_pose_,false);
    //将当前 submap_cloud_ 加入到 容器中 ， 攒一波大的 ， 开多线程同时对多个submap的点云进行处理
    if (submap_collect.size() > 15 || (cloud_file_ind == cloud_files.size() - 1 && submap_collect.size() > 0)) {
      vector<int> sub_inds;
      std::vector<future<bool>> futures; //! multi thread 
      utils::TicToc tic_mt;
      for(volatile int sub_ind = 0;sub_ind < submap_collect.size();sub_ind++) {
        //! single thread 
        // if(post_process(submap_collect[sub_ind].cloud_ptr)) {
        //   sub_inds.push_back(sub_ind);
        //   LOG(INFO) << "process sub map.";
        // }
        // else {
        //   LOG(INFO) << "post process failed.";
        // }

        //! multi thread 
        futures.push_back(std::async(&Submap_generator::post_process,this,submap_collect[sub_ind].cloud_ptr));
      }

      //! multi thread  start
      int future_ind = 0;
      for(auto& future : futures) {
        volatile bool res = future.get();
        if(res) {
          sub_inds.push_back(future_ind);
          // LOG(INFO) << "thread : " << future_ind << "  res :  " << res;
        }
        future_ind++;
      }
      //! multi thread  end 
      double mt_time = tic_mt.toc();
      LOG(INFO) << "multi-thread post processing time used : " << mt_time;
      

      for(int ind : sub_inds) {
        save_submap(submap_collect[ind]);
      }
      submap_collect.clear();
    }
    
    LOG(INFO) << "generate submap success.";
    //删除部分当前 cloud_data_q 中的过时的数据
    remove_old_cloud_data();
  }
  return true;
}

Cloud_data Submap_generator::get_cloud_data_from_file(const Cloud_file _cloud_file,const Matrix4f current_cloud_pose) {
  /*
    读取点云bin 文件并 将 其转换到imu 系中
  */
  PointCloudPtr cloud_(new PointCloud);
  utils::load_kaist_cloud(_cloud_file.file_path,cloud_);

  if(_cloud_file.is_ns1) {
    // pcl::transformPointCloud(*cloud_,*cloud_,T_ns1_imu);
    pcl::transformPointCloud(*cloud_,*cloud_,T_ego_ns1);
  }
  else{
    // pcl::transformPointCloud(*cloud_,*cloud_,T_ns2_imu);
    pcl::transformPointCloud(*cloud_,*cloud_,T_ego_ns2);
  }
  return Cloud_data(cloud_,_cloud_file.stamp,current_cloud_pose,_cloud_file.is_ns1);
}

//load data into stamp_gt_pose_vec
bool Submap_generator::load_global_pose(const string _global_pose_path) {
  ifstream pose_file(_global_pose_path);
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
    Matrix4f pose;
    pose << row[1] , row[2] , row[3] , row[4],
            row[5] , row[6] , row[7] , row[8],
            row[9] , row[10], row[11], row[12],
            0      ,       0,       0,   1;
    stamp_gt_poses.emplace_back(stamp,pose);
  }
  gt_pose_offset = stamp_gt_poses[0].pose.block<3,1>(0,3);
  for(int ind = stamp_gt_poses.size() - 1;ind >= 0;ind--) {
    stamp_gt_poses[ind].pose.block<3,1>(0,3) -= gt_pose_offset;
  }
  return true;
}

bool Submap_generator::gen_submap(PointCloudPtr _submap) {
  /*
    1. 根据yaml 文件中设置的 使用 n 帧 拼接submap , 
    2.生成  s * s * s大小的submap 
    3. 对submap 内点云进行地面滤波 ， voxel grid降采样 ， fps降采样到 p 个点
  */
  PointCloudPtr cloud_submap(new PointCloud());
  if (cloud_data_q.size() < submap_frame_num) {
    LOG(ERROR) << "not enough kf in q,gen submap failed.";
    return false;
  }
  
  //check submap gap 
  if((cloud_data_q[submap_frame_num / 2].pose_stamp.pose.block<3,1>(0,3) - 
      last_submap_pose).norm() < submap_gap) {
    return false;
  }
  

  for(int i = 0;i < submap_frame_num;i++) {
    PointCloudPtr kf_global(new PointCloud());
    pcl::transformPointCloud(*cloud_data_q[i].cloud_ptr,*kf_global,cloud_data_q[i].pose_stamp.pose);
    *cloud_submap += *kf_global;
  }
  last_submap_pose = cloud_data_q[submap_frame_num / 2].pose_stamp.pose.block<3,1>(0,3);
  last_submap_stamp = cloud_data_q[submap_frame_num / 2].pose_stamp.stamp;
  *_submap = *cloud_submap;
  
  //  将当前世界系下的点云转换到中间帧的lidar系下。  在转换到ego frame
  Eigen::Matrix4f T_ego_lidar = cloud_data_q[submap_frame_num / 2].is_ns1 ? T_ego_ns1 :T_ego_ns2;
  Eigen::Matrix4f T_global_ego = T_ego_lidar * cloud_data_q[submap_frame_num / 2].pose_stamp.pose.inverse();
  pcl::transformPointCloud(*_submap,*_submap,(cloud_data_q[submap_frame_num / 2].pose_stamp.pose).inverse());
  return true;
}

bool Submap_generator::get_pose_from_stamp(const double _stamp,int& _ind_in_q,Eigen::Matrix4f& _p) {
  if (stamp_gt_poses.empty()) {
    LOG(INFO) << "No gt pose in deque.";
    return false;
  }
  
  if(_stamp > stamp_gt_poses.back().stamp) {
    return false;
  }

  int ind = 0;
  while(ind < stamp_gt_poses.size() && _stamp > stamp_gt_poses[ind].stamp) {
    ind ++;
  }
  if(ind == 0 || ind == stamp_gt_poses.size()) {
    return false;
  }

  Eigen::Matrix4f prev_pose = stamp_gt_poses[ind - 1].pose;
  Eigen::Matrix4f next_pose = stamp_gt_poses[ind].pose;
  Eigen::Matrix4f frame_pose = Eigen::Matrix4f::Identity();
  double prev_ratio = (stamp_gt_poses[ind].stamp - _stamp) / (stamp_gt_poses[ind].stamp - stamp_gt_poses[ind - 1].stamp);
  if(!utils::interpolate_pose(prev_pose,next_pose,prev_ratio,frame_pose)) {
    LOG(INFO) << "Interpolating pose failed.";
    return false;
  }
  
  _ind_in_q = ind - 1;
  _p = frame_pose;
  return true;
}

bool Submap_generator::check_if_key_frame(const Matrix4f _prev_key_frame_pose,const Matrix4f _current_pose) {
  Vector3f prev_pos = _prev_key_frame_pose.block<3,1>(0,3);
  Vector3f curr_pos = _current_pose.block<3,1>(0,3);
  Vector3f diff = curr_pos - prev_pos;
  return diff.norm() > 0.4 ? true: false;
}

bool Submap_generator::post_process(PointCloudPtr _submap_cloud) {
  /*
    1. tranform to zero , cal xmax ymax xmin ymin 
    2. gournd removal 
    3. clip submap 
    4. voxel grid 
    5. fps 
    6. coord normalization [-1,1]
  */
  //TODO 先将所有的点云提升1.5 m 
  for(int i = 0;i < _submap_cloud->size();i++) {
    _submap_cloud->points[i].z -= 1.5;
  }
  PointCloudPtr non_ground_cloud(new PointCloud);
  for(int i = 0;i < _submap_cloud->size();i++) {
    if(abs(_submap_cloud->points[i].z + 1.5) > 0.5) 
    {
      non_ground_cloud->push_back(_submap_cloud->points[i]);
    }
  }
  *_submap_cloud = *non_ground_cloud;
  //! 由于当前直接就是在 ego 系下的点云直接使用 直接对原始的ego 系下的点云进行地面滤波
  // PointCloudPtr non_ground_cloud(new PointCloud);
  // PointCloudPtr ground_cloud(new PointCloud);
  // utils::Ground_filter<Point> gf_(node);
  // gf_.ground_filtering(_submap_cloud,ground_cloud,non_ground_cloud);
  // *_submap_cloud = *non_ground_cloud;
  

  // double xmin = std::numeric_limits<double>::max();
  // double ymin = std::numeric_limits<double>::max();
  // double xmax = std::numeric_limits<double>::min();
  // double ymax = std::numeric_limits<double>::min();
  // double zmax = std::numeric_limits<double>::min();
  // double zmin = std::numeric_limits<double>::max();
  // //计算所有点云的 xy 质心 ， 
  // Eigen::Vector4f centroid;					// 质心
  // pcl::compute3DCentroid(*_submap_cloud, centroid);	// 齐次坐标，（c0,c1,c2,1）
  // for(int p_ind = 0;p_ind < _submap_cloud->size();p_ind++) {
  //   if(_submap_cloud->points[p_ind].x < xmin) {
  //     xmin = _submap_cloud->points[p_ind].x;
  //   }
  //   if(_submap_cloud->points[p_ind].y < ymin) {
  //     ymin = _submap_cloud->points[p_ind].y;
  //   }
  //   if(_submap_cloud->points[p_ind].z < zmin) {
  //     zmin = _submap_cloud->points[p_ind].z;
  //   }
  //   if(_submap_cloud->points[p_ind].x > xmax) {
  //     xmax = _submap_cloud->points[p_ind].x;
  //   }
  //   if(_submap_cloud->points[p_ind].y > ymax) {
  //     ymax = _submap_cloud->points[p_ind].y;
  //   }
  //   if(_submap_cloud->points[p_ind].z > zmax) {
  //     zmax = _submap_cloud->points[p_ind].z;
  //   }
  // }
  // double x_range = xmax - xmin;
  // double y_range = ymax - ymin;
  // double z_range = zmax - zmin;
  // Vector3f offset((xmin + xmax) /2. , (ymin + ymax) /2.,zmin);
  
  // for(int p_ind = 0;p_ind < _submap_cloud->size();p_ind++) {
  //   _submap_cloud->points[p_ind].x -= centroid.x();
  //   _submap_cloud->points[p_ind].y -= centroid.y();
  //   _submap_cloud->points[p_ind].z -= centroid.z(); 
  // }

  double half_submap_cube_size = submap_cube_size / 2.;
  pcl::PassThrough<Point> pass_x; // 声明直通滤波
	pass_x.setInputCloud(_submap_cloud); // 传入点云数据
	pass_x.setFilterFieldName("x"); // 设置操作的坐标轴
	pass_x.setFilterLimits(-half_submap_cube_size, half_submap_cube_size); // 设置坐标范围
	// pass.setFilterLimitsNegative(true); // 保留数据函数
	pass_x.filter(*_submap_cloud);  // 进行滤波输出

  pcl::PassThrough<Point> pass_y; // 声明直通滤波
	pass_x.setInputCloud(_submap_cloud); // 传入点云数据
	pass_x.setFilterFieldName("y"); // 设置操作的坐标轴
	pass_x.setFilterLimits(-half_submap_cube_size, half_submap_cube_size); // 设置坐标范围
	// pass.setFilterLimitsNegative(true); // 保留数据函数
	pass_x.filter(*_submap_cloud);  // 进行滤波输出


  pcl::PassThrough<Point> pass_z; // 声明直通滤波
	pass_x.setInputCloud(_submap_cloud); // 传入点云数据
	pass_x.setFilterFieldName("z"); // 设置操作的坐标轴
	pass_x.setFilterLimits(-30, 30.0); // 设置坐标范围
	// pass.setFilterLimitsNegative(true); // 保留数据函数
	pass_x.filter(*_submap_cloud);  // 进行滤波输出


  // //! 使用地面滤波
  // PointCloudPtr non_ground_cloud(new PointCloud);
  // utils::ground_filter(_submap_cloud,non_ground_cloud);
  // *_submap_cloud = *non_ground_cloud;


  // LOG(INFO) << "non ground cloud size : " << _submap_cloud->size();
  //进行 voxel grid 的降采样
  if(_submap_cloud->size() > 150000) {
    pcl::VoxelGrid<Point> sor;
    sor.setInputCloud (_submap_cloud);
    sor.setLeafSize (0.3f, 0.3f, 0.3f);
    sor.filter (*_submap_cloud);
  }
  else if (_submap_cloud->size() > 100000){
    pcl::VoxelGrid<Point> sor;
    sor.setInputCloud (_submap_cloud);
    sor.setLeafSize (0.1f, 0.1f, 0.1f);
    sor.filter (*_submap_cloud);
  }
  if(_submap_cloud->size() > 30000) {
    pcl::RandomSample<Point> rd;
    rd.setSample(20000);
    rd.setInputCloud(_submap_cloud);
    rd.filter(*_submap_cloud);
  }


  //FPS 
  utils::TicToc fps_tic;
  vector<int> sampled_index;
  if(!fps(_submap_cloud,submap_point_num,sampled_index)) {
    LOG(ERROR) << "fps failed.raw submap poinst is not enougn for fps.";
    return false;
  }
  PointCloudPtr fps_cloud(new PointCloud());
  for(int sample_ind = 0;sample_ind < sampled_index.size();sample_ind++) {
    fps_cloud->push_back(_submap_cloud->points[sampled_index[sample_ind]]);
  }
  *_submap_cloud = *fps_cloud;
  double fps_time_used = fps_tic.toc();
  // LOG(INFO) << "fps time used : " << fps_time_used;


  //将所有的点云的坐标进行归一化。 保证所有的点云都是 zero mean ， 以及坐标的范围在 [-1,1]之间
  // double ratio_ = (std::max(x_range/2,y_range /2),std::max(x_range/2,z_range /2));
  double ratio_ = half_submap_cube_size;
  for(int p_ind = 0;p_ind < _submap_cloud->points.size();p_ind++) {
    _submap_cloud->points[p_ind].x /= ratio_;
    _submap_cloud->points[p_ind].y /= ratio_;
    _submap_cloud->points[p_ind].z /= ratio_;
  }

  return true;
}

bool Submap_generator::save_submap(PointCloudPtr _submap_cloud) {
  //将当前的 submap_cloud 保存在 data目录下
  
  std::string save_path = cloud_save_dir + "/" + to_string(last_submap_stamp) + ".pcd";
  pcl::io::savePCDFile(save_path,*_submap_cloud);
  LOG(INFO) << "submap saved at : " << save_path;

  //还需要保存一个 csv 将 当前 submap 的位置 时间戳记录下来
  Vector3f sub_map_gt = last_submap_pose + gt_pose_offset;
  of_stamp_pose << setprecision(18) << to_string(last_submap_stamp) << " ";
  of_stamp_pose << sub_map_gt.x() << " " << sub_map_gt.y() << " " << sub_map_gt.z() << endl;
}

__attribute__((optimize("O0"))) bool Submap_generator::save_submap(Cloud_data& _cloud_data) {
  std::string save_path = cloud_save_dir + "/" + to_string(_cloud_data.pose_stamp.stamp) + ".pcd";
  pcl::io::savePCDFile(save_path,*_cloud_data.cloud_ptr);
  LOG(INFO) << "submap saved at : " << save_path;

  //还需要保存一个 csv 将 当前 submap 的位置 时间戳记录下来
  Vector3f sub_map_gt = _cloud_data.pose_stamp.pose.block<3,1>(0,3) + gt_pose_offset;
  of_stamp_pose << setprecision(18) << to_string(_cloud_data.pose_stamp.stamp) << " ";
  of_stamp_pose << sub_map_gt.x() << " " << sub_map_gt.y() << " " << sub_map_gt.z() << endl;
  LOG(INFO) << "4";
}

std::string Submap_generator::extractFilename(const std::string& path)
{
    // 查找最后一个路径分隔符的位置
    size_t lastSeparator = path.find_last_of("/\\");

    // 提取文件名子字符串
    std::string filename = path.substr(lastSeparator + 1);

    // 移除文件扩展名
    size_t dotPos = filename.find_last_of(".");
    if (dotPos != std::string::npos)
    {
        filename = filename.substr(0, dotPos);
    }

    return filename;
}

void Submap_generator::remove_old_cloud_data() {
  int max_frame_num = 2*submap_frame_num;
  while(cloud_data_q.size() > max_frame_num) {
    cloud_data_q.pop_back();
  }
}


bool Submap_generator::fps(PointCloudPtr _cloud_in,int _target_num,vector<int>& _sample_index) {
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

void Submap_generator::graduation_thesis_vis() {
  /*
    1. 根据当前seq的地址 ， 读取所有的 right_vlp , left vlp 中的 bin文件 
      stamp -> (ns1 or ns2 , abs_path)
    2. 将所有的点云的 加入到一个队列中 ， 依次弹出 ，
        2.1 trans to imu frame 
        2.2 query pose -> trans to global frame
        2.3 if current pose key frame （ns1 or ns2） add into cloud_data_q
        2.4 if current frame pose differ from last_submap_pose 
            generate new submap 
            save submap 
  */

  //1. load point cloud bin file , generate vector & sort it by stamp 
  vector<Cloud_file> cloud_files;
  std::string ns1_cloud_dir = current_seq_data_root + "sensor_data/VLP_right/";
  std::string ns2_cloud_dir = current_seq_data_root + "sensor_data/VLP_left/";

  //检查 两个文件夹是否存在
  if (!std::filesystem::exists(std::filesystem::path(ns1_cloud_dir))) {
    LOG(ERROR) << ns1_cloud_dir << " not exist. skip.";
  }
  if (!std::filesystem::exists(std::filesystem::path(ns2_cloud_dir))) {
    LOG(ERROR) << ns2_cloud_dir << " not exist. skip.";
  }

  //将两个文件夹中的文件全部 加入到 cloud_files中
  for (const auto& entry : std::filesystem::directory_iterator(ns1_cloud_dir)){
    string abs_file_path_ = entry.path();
    double stamp_ = stod(extractFilename(abs_file_path_)) * 1e-9;
    cloud_files.emplace_back(abs_file_path_,true,stamp_);
  }
  for (const auto& entry : std::filesystem::directory_iterator(ns2_cloud_dir)){
    string abs_file_path_ = entry.path();
    double stamp_ = stod(extractFilename(abs_file_path_)) * 1e-9;
    cloud_files.emplace_back(abs_file_path_,false,stamp_); 
  }
  std::sort(cloud_files.begin(), cloud_files.end(), compare_cloud_file); //根据时间戳 对 cloud_files进行排序
  
  // 2 for loop for every cloud
  for(int cloud_file_ind = 0;cloud_file_ind < cloud_files.size();cloud_file_ind ++) {
    LOG(INFO) << current_seq << "  processing " << cloud_file_ind << "  /  " << cloud_files.size();
    Cloud_file current_cloud_file = cloud_files[cloud_file_ind];
    Matrix4f current_cloud_pose = Matrix4f::Identity();
    
    int stamp_ind;  //not used 
    if(!get_pose_from_stamp(current_cloud_file.stamp,stamp_ind,current_cloud_pose)) {
      continue;
    }
    // LOG(INFO) << "current_cloud_pose = \n" << current_cloud_pose;
    if (current_cloud_file.is_ns1) {
      /*
        当前文件 ns1 
      */
      if (check_if_key_frame(prev_ns1_kf_pose,current_cloud_pose)) {
        prev_ns1_kf_pose = current_cloud_pose;
      }
      else {
        continue;
      }
    }
    else {
      /*
        ns2  
      */
      //当前帧是关键帧 进行处理
      if (check_if_key_frame(prev_ns2_kf_pose,current_cloud_pose)) {
        prev_ns2_kf_pose = current_cloud_pose;
      }
      else {
        //如果当前帧不是关键帧 就不进行处理了
        continue;
      }
    }
    
    // add into cloud_data_q 
    cloud_data_q.push_front(get_cloud_data_from_file(current_cloud_file,current_cloud_pose));
    // LOG(INFO) << "current cloud size = " << cloud_data_q[0].cloud_ptr->size();
    //可能还 需要修改一下 submap_cloud_ 关于 last_submap_pose & stamp 的一些设置方法。
    // 好像不用改。。。。。
    PointCloudPtr submap_cloud_(new PointCloud);
    //这里生成的是原始的点云地图，没有经过任何处理
    if(!gen_submap(submap_cloud_)) {
      continue;
    }

    static int count = 0;
    string out_dir = "/home/yyj/catkin_ws/src/kaist_yyj/graduation_thesis";
    string file_name = out_dir + "/submap_" + std::to_string(count++) + ".pcd";
    pcl::io::savePCDFile(file_name,*submap_cloud_);
    LOG(INFO) << file_name + " saved.";
  PointCloudPtr _submap_cloud = submap_cloud_;
  for(int i = 0;i < _submap_cloud->size();i++) {
    _submap_cloud->points[i].z -= 1.5;
  }
  PointCloudPtr non_ground_cloud(new PointCloud);
  for(int i = 0;i < _submap_cloud->size();i++) {
    if(abs(_submap_cloud->points[i].z + 1.5) > 0.5) 
    {
      non_ground_cloud->push_back(_submap_cloud->points[i]);
    }
  }
  *_submap_cloud = *non_ground_cloud;

  //TODO save non ground cloud 
    file_name = out_dir + "/submap_" + std::to_string(count++) + "_nonground1.pcd";
    pcl::io::savePCDFile(file_name,*_submap_cloud );
    LOG(INFO) << file_name + " saved.";

  //! 由于当前直接就是在 ego 系下的点云直接使用 直接对原始的ego 系下的点云进行地面滤

  //不进行地面滤波了 使用 直通滤波 对 xy 维度进行滤波'
  double half_submap_cube_size = submap_cube_size / 2.;
  pcl::PassThrough<Point> pass_x; // 声明直通滤波
	pass_x.setInputCloud(_submap_cloud); // 传入点云数据
	pass_x.setFilterFieldName("x"); // 设置操作的坐标轴
	pass_x.setFilterLimits(-half_submap_cube_size, half_submap_cube_size); // 设置坐标范围
	// pass.setFilterLimitsNegative(true); // 保留数据函数
	pass_x.filter(*_submap_cloud);  // 进行滤波输出

  pcl::PassThrough<Point> pass_y; // 声明直通滤波
	pass_x.setInputCloud(_submap_cloud); // 传入点云数据
	pass_x.setFilterFieldName("y"); // 设置操作的坐标轴
	pass_x.setFilterLimits(-half_submap_cube_size, half_submap_cube_size); // 设置坐标范围
	// pass.setFilterLimitsNegative(true); // 保留数据函数
	pass_x.filter(*_submap_cloud);  // 进行滤波输出


  pcl::PassThrough<Point> pass_z; // 声明直通滤波
	pass_x.setInputCloud(_submap_cloud); // 传入点云数据
	pass_x.setFilterFieldName("z"); // 设置操作的坐标轴
	pass_x.setFilterLimits(-30, 30.0); // 设置坐标范围
	// pass.setFilterLimitsNegative(true); // 保留数据函数
	pass_x.filter(*_submap_cloud);  // 进行滤波输出

  
  //TODO save pass through filtered cloud
  file_name = out_dir + "/submap_" + std::to_string(count++) + "_passthrough2.pcd";
    pcl::io::savePCDFile(file_name,*_submap_cloud );
    LOG(INFO) << file_name + " saved.";


  //进行 voxel grid 的降采样
  if(_submap_cloud->size() > 150000) {
    pcl::VoxelGrid<Point> sor;
    sor.setInputCloud (_submap_cloud);
    sor.setLeafSize (0.3f, 0.3f, 0.3f);
    sor.filter (*_submap_cloud);
  }
  else if (_submap_cloud->size() > 100000){
    pcl::VoxelGrid<Point> sor;
    sor.setInputCloud (_submap_cloud);
    sor.setLeafSize (0.1f, 0.1f, 0.1f);
    sor.filter (*_submap_cloud);
  }
  
  //TODO save voxel grid filtered cloud
  file_name = out_dir + "/submap_" + std::to_string(count++) + "_voxelgrid3.pcd";
    pcl::io::savePCDFile(file_name,*_submap_cloud );
    LOG(INFO) << file_name + " saved.";
  if(_submap_cloud->size() > 30000) {
    pcl::RandomSample<Point> rd;
    rd.setSample(20000);
    rd.setInputCloud(_submap_cloud);
    rd.filter(*_submap_cloud);
  }

  //TODO save random sample filtered cloud
  file_name = out_dir + "/submap_" + std::to_string(count++) + "_randomsample4.pcd";
    pcl::io::savePCDFile(file_name,*_submap_cloud );
    LOG(INFO) << file_name + " saved.";

  //FPS 
  utils::TicToc fps_tic;
  vector<int> sampled_index;
  if(!fps(_submap_cloud,submap_point_num,sampled_index)) {
    LOG(ERROR) << "fps failed.raw submap poinst is not enougn for fps.";
    return;
  }
  PointCloudPtr fps_cloud(new PointCloud());
  for(int sample_ind = 0;sample_ind < sampled_index.size();sample_ind++) {
    fps_cloud->push_back(_submap_cloud->points[sampled_index[sample_ind]]);
  }
  *_submap_cloud = *fps_cloud;
  double fps_time_used = fps_tic.toc();
  // LOG(INFO) << "fps time used : " << fps_time_used;
  //TODO save fps filtered cloud
  file_name = out_dir + "/submap_" + std::to_string(count++) + "_fps5.pcd";
    pcl::io::savePCDFile(file_name,*_submap_cloud );
    LOG(INFO) << file_name + " saved.";
  //将所有的点云的坐标进行归一化。 保证所有的点云都是 zero mean ， 以及坐标的范围在 [-1,1]之间
  // double ratio_ = (std::max(x_range/2,y_range /2),std::max(x_range/2,z_range /2));
  double ratio_ = half_submap_cube_size;
  for(int p_ind = 0;p_ind < _submap_cloud->points.size();p_ind++) {
    _submap_cloud->points[p_ind].x /= ratio_;
    _submap_cloud->points[p_ind].y /= ratio_;
    _submap_cloud->points[p_ind].z /= ratio_;
  }
    //TODO save rescaled cloud
    file_name = out_dir + "/submap_" + std::to_string(count++) + "_rescale6.pcd";
    pcl::io::savePCDFile(file_name,*_submap_cloud );
    LOG(INFO) << file_name + " saved.";
    LOG(INFO) << "generate submap success.";
    //删除部分当前 cloud_data_q 中的过时的数据
    remove_old_cloud_data();
  }
}