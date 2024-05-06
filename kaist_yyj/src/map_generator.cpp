#include <kaist_yyj/map_generator.h>

Map_generator::Map_generator(const ros::NodeHandle& _nh,const YAML::Node& _node) {
  nh = _nh;
  node = _node;

  sub_cloud_ns1 = nh.subscribe("/ns1/velodyne_points",10000000,&Map_generator::ns1_cloud_cb,this);
  sub_cloud_ns2 = nh.subscribe("/ns2/velodyne_points",10000000,&Map_generator::ns2_cloud_cb,this);
  pub_submap_glo = nh.advertise<sensor_msgs::PointCloud2>("submap_glo",100);
  
  current_seq           = _node["current_seq"].as<string>();   //"urban17"
  seq_data_root         = _node["seq_data_root"].as<string>(); ///home/yyj/download/kaist/
  current_seq_data_root = seq_data_root + current_seq + "/";
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
  
  //Load global pose txt 
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

//load data into stamp_gt_pose_vec
bool Map_generator::load_global_pose(const string _global_pose_path) {
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

//just add cloud into deque and ask for pose
void Map_generator::ns1_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& _msg_in) {
    /*
    0. transform cloud to imu frame
    1. ask for pose 
    2. check if current frame ket frame 
        y: add current_frame(global) into q
        n: skip 
    */
  // LOG(INFO) << "ns1 cloud msg recieved.";
  PointCloudPtr curr_cloud(new PointCloud());
  pcl::fromROSMsg(*_msg_in,*curr_cloud);
  pcl::transformPointCloud(*curr_cloud,*curr_cloud,T_ns1_imu);
  // 1  query pose of frame
  double frame_stamp = _msg_in->header.stamp.toSec();
  int ind_q;
  Eigen::Matrix4f frame_gt_pose = Eigen::Matrix4f::Identity();
  if(!get_pose_from_stamp(frame_stamp,ind_q,frame_gt_pose)) {
    LOG(INFO) << setprecision(18) << "No pose for current frame, stamp : " << frame_stamp;
    return;
  }

  // 2 check if key frame  , add kf into q if key frame
  if (!check_if_key_frame(prev_ns1_kf_pose,frame_gt_pose)) {
    return;
  }
  prev_ns1_kf_pose = frame_gt_pose;
  cloud_data_q.emplace_front(curr_cloud,frame_stamp,frame_gt_pose);
}

// trigger
void Map_generator::ns2_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& _msg_in) {
    /*
    0. transform cloud to imu frame
    1. ask for pose 
    2. check if current frame ket frame 
        y: add current_frame(global) into q
        n: skip 
    3. if current frame is far from last submap :
        generate new submap
    */
  // LOG(INFO) << "ns2 cloud msg recieved.";
  PointCloudPtr curr_cloud(new PointCloud());
  pcl::fromROSMsg(*_msg_in,*curr_cloud);
  pcl::transformPointCloud(*curr_cloud,*curr_cloud,T_ns2_imu);
  // 1  query pose of frame
  double frame_stamp = _msg_in->header.stamp.toSec();
  int ind_q;
  Eigen::Matrix4f frame_gt_pose = Eigen::Matrix4f::Identity();
  if(!get_pose_from_stamp(frame_stamp,ind_q,frame_gt_pose)) {
    LOG(INFO) << setprecision(18) << "No pose for current frame, stamp : " << frame_stamp;
    return;
  }

  // 2 check if key frame  , add kf into q if key frame
  if (!check_if_key_frame(prev_ns2_kf_pose,frame_gt_pose)) {
    LOG(INFO) << "not kf . skip .";
    return;
  }
  prev_ns2_kf_pose = frame_gt_pose;

  cloud_data_q.emplace_front(curr_cloud,frame_stamp,frame_gt_pose);

  //3 trigger seg submap 
  
  utils::TicToc tic;
  PointCloudPtr submap_cloud(new PointCloud());
  //如果当前地图的中心较上一个地图的 中心距离小于 阈值 ， 则不生成地图
  if (!gen_submap(submap_cloud)) {
    // LOG(WARNING) << "gen submap failed.";
    return;
  }

    /*
    TODO 保存 submap
    submap 后处理： 1. 中心归零(平移) 2.ground removal 3. voxel grid 4. fps 
    txt :   stamp * 1e9（filename） & pose (add offset)
    bin_file : 
  */
  post_process(submap_cloud);
  save_submap(submap_cloud); //TODO 

  sensor_msgs::PointCloud2 submap_msg;
  pcl::toROSMsg(*submap_cloud,submap_msg);
  submap_msg.header.frame_id = "global";
  submap_msg.header.stamp = _msg_in->header.stamp;
  pub_submap_glo.publish(submap_msg);
  LOG(INFO) << "gen submap sucessed.";
  LOG(INFO) << "submap size = " <<submap_cloud->size();
  //及时删除一些历史的关键帧
  int max_frame_num = 2*submap_frame_num;
  while(cloud_data_q.size() > max_frame_num) {
    cloud_data_q.pop_back();
  }

  double time_used = tic.toc();
  LOG(INFO) << "gen submap time used : " << time_used;
}

bool Map_generator::gen_submap(PointCloudPtr _submap) {
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

  return true;
}

bool Map_generator::get_pose_from_stamp(const double _stamp,int& _ind_in_q,Eigen::Matrix4f& _p) {
  if (stamp_gt_poses.empty()) {
    LOG(INFO) << "No gt pose in deque.";
    return false;
  }

  int ind = 0;
  while(_stamp > stamp_gt_poses[ind].stamp) {
    ind ++;
  }
  if(ind == 0) {
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

bool Map_generator::check_if_key_frame(const Matrix4f _prev_key_frame_pose,const Matrix4f _current_pose) {
  Vector3f prev_pos = _prev_key_frame_pose.block<3,1>(0,3);
  Vector3f curr_pos = _current_pose.block<3,1>(0,3);
  Vector3f diff = curr_pos - prev_pos;
  return diff.norm() > 0.4 ? true: false;
}

bool Map_generator::post_process(PointCloudPtr _submap_cloud) {
  /*
    1. tranform to zero , cal xmax ymax xmin ymin 
    2. gournd removal 
    3. clip submap 
    4. voxel grid 
    5. fps 
    6. coord normalization [-1,1]
  */
  double xmin = std::numeric_limits<double>::max();
  double ymin = std::numeric_limits<double>::max();
  double xmax = std::numeric_limits<double>::min();
  double ymax = std::numeric_limits<double>::min();
  double zmax = std::numeric_limits<double>::min();
  double zmin = std::numeric_limits<double>::max();
  //计算所有点云的 xy 质心 ， 
  Eigen::Vector4f centroid;					// 质心
  pcl::compute3DCentroid(*_submap_cloud, centroid);	// 齐次坐标，（c0,c1,c2,1）
  for(int p_ind = 0;p_ind < _submap_cloud->size();p_ind++) {
    if(_submap_cloud->points[p_ind].x < xmin) {
      xmin = _submap_cloud->points[p_ind].x;
    }
    if(_submap_cloud->points[p_ind].y < ymin) {
      ymin = _submap_cloud->points[p_ind].y;
    }
    if(_submap_cloud->points[p_ind].z < zmin) {
      zmin = _submap_cloud->points[p_ind].z;
    }
    if(_submap_cloud->points[p_ind].x > xmax) {
      xmax = _submap_cloud->points[p_ind].x;
    }
    if(_submap_cloud->points[p_ind].y > ymax) {
      ymax = _submap_cloud->points[p_ind].y;
    }
    if(_submap_cloud->points[p_ind].z > zmax) {
      zmax = _submap_cloud->points[p_ind].z;
    }
  }
  double x_range = xmax - xmin;
  double y_range = ymax - ymin;
  double z_range = zmax - zmin;
  // Vector3f offset((xmin + xmax) /2. , (ymin + ymax) /2.,zmin);
  
  for(int p_ind = 0;p_ind < _submap_cloud->size();p_ind++) {
    _submap_cloud->points[p_ind].x -= centroid.x();
    _submap_cloud->points[p_ind].y -= centroid.y();
    _submap_cloud->points[p_ind].z -= centroid.z(); 
  }

  //不进行地面滤波了 使用 直通滤波 对 xy 维度进行滤波
  pcl::PassThrough<Point> pass_x; // 声明直通滤波
	pass_x.setInputCloud(_submap_cloud); // 传入点云数据
	pass_x.setFilterFieldName("x"); // 设置操作的坐标轴
	pass_x.setFilterLimits(-30, 30.0); // 设置坐标范围
	// pass.setFilterLimitsNegative(true); // 保留数据函数
	pass_x.filter(*_submap_cloud);  // 进行滤波输出

  pcl::PassThrough<Point> pass_y; // 声明直通滤波
	pass_x.setInputCloud(_submap_cloud); // 传入点云数据
	pass_x.setFilterFieldName("y"); // 设置操作的坐标轴
	pass_x.setFilterLimits(-30, 30.0); // 设置坐标范围
	// pass.setFilterLimitsNegative(true); // 保留数据函数
	pass_x.filter(*_submap_cloud);  // 进行滤波输出


  //进行 voxel grid 的降采样
  pcl::VoxelGrid<Point> sor;
  sor.setInputCloud (_submap_cloud);
  sor.setLeafSize (0.5f, 0.5f, 0.5f);
  sor.filter (*_submap_cloud);
  LOG(INFO) << "downsampled cloud size : " << _submap_cloud->size();
  

  // PointCloudPtr ground_cloud(new PointCloud);
  // PointCloudPtr non_ground_cloud(new PointCloud);
  // ground_filter_ptr->ground_filtering(_submap_cloud,ground_cloud,non_ground_cloud);
  // *_submap_cloud = *non_ground_cloud;
  
  //FPS 
  utils::TicToc fps_tic;
  vector<int> sampled_index;
  if(!utils::fps(_submap_cloud,submap_point_num,sampled_index)) {
    LOG(ERROR) << "fps failed.raw submap poinst is not enougn for fps.";
    return false;
  }
  PointCloudPtr fps_cloud(new PointCloud());
  for(int sample_ind = 0;sample_ind < sampled_index.size();sample_ind++) {
    fps_cloud->push_back(_submap_cloud->points[sampled_index[sample_ind]]);
  }
  *_submap_cloud = *fps_cloud;
  double fps_time_used = fps_tic.toc();
  LOG(INFO) << "fps time used : " << fps_time_used;


  //将所有的点云的坐标进行归一化。 保证所有的点云都是 zero mean ， 以及坐标的范围在 [-1,1]之间
  #ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for(int p_ind = 0;p_ind < _submap_cloud->points.size();p_ind++) {
    _submap_cloud->points[p_ind].x /= (x_range/2);
    _submap_cloud->points[p_ind].y /= (y_range/2);
    _submap_cloud->points[p_ind].z /= (z_range/2);
  }

  return true;
}

bool Map_generator::save_submap(PointCloudPtr _submap_cloud) {
  //将当前的 submap_cloud 保存在 data目录下
  
  std::string save_path = cloud_save_dir + "/" + to_string(last_submap_stamp) + ".pcd";
  pcl::io::savePCDFile(save_path,*_submap_cloud);
  LOG(INFO) << "submap saved at : " << save_path;

  //还需要保存一个 csv 将 当前 submap 的位置 时间戳记录下来
  Vector3f sub_map_gt = last_submap_pose + gt_pose_offset;
  of_stamp_pose << setprecision(18) << to_string(last_submap_stamp) << " ";
  of_stamp_pose << sub_map_gt.x() << " " << sub_map_gt.y() << " " << sub_map_gt.z() << endl;
}