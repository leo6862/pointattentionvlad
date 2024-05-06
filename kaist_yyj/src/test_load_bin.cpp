// #include <iostream>
// #include <fstream>
// #include <vector>

// #include <pcl/io/pcd_io.h>
// #include <pcl/io/file_io.h>
// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>

// using namespace std;

// int main()
// {
//     // Create a Point Cloud object
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

//     // Load the .bin file
//     std::string filename = "/home/yyj/download/kaist/urban17/sensor_data/VLP_left/1524211202189730000.bin";
    
//     std::ifstream file(filename,ios::in|ios::binary|ios::ate);
//     streampos size;
//     char * memblock;
//     size = file.tellg();

//     cout << "size=" << size << "\n"; 

//     memblock = new char [size];
//     file.seekg (0, ios::beg);
//     file.read (memblock, size);
//     file.close();

//     cout << "the entire file content is in memory \n";
//     double* double_values = (double*)memblock;//reinterpret as doubles
//     int i = 0;
//     for(i=0; i<=10; i++)
//     {
//         double value = double_values[i];
//         cout << "value ("<<i<<")=" << value << "\n";
//     }


//     // Access the loaded point cloud data
//     for (const auto& point : cloud->points) {
//         std::cout << "x: " << point.x << ", y: " << point.y << ", z: " << point.z << std::endl;
//     }

//     return 0;
// }



#include <iostream>
#include <fstream>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

struct PointXYZIT
{
    float x;
    float y;
    float z;
    float intensity;
    // uint32_t timestamp;
};

int main()
{
    std::string filename = "/home/yyj/download/kaist/urban14/sensor_data/VLP_left/1524189302278919000.bin";

    // Open the binary file
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    // Read the binary data into a vector of PointXYZIT
    std::vector<PointXYZIT> pointCloud;
    PointXYZIT point;
    while (file.read((char*)&point, sizeof(PointXYZIT)))
    {
        pointCloud.push_back(point);
    }

    std::cout << "cloud size = " << pointCloud.size();
    // Close the file
    file.close();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // Access and process the loaded point cloud data
    for (const auto& point : pointCloud)
    {
        // Access point.x, point.y, point.z, point.intensity, point.timestamp
        // ...
        std::cout << "x: " << point.x << ", y: " << point.y << ", z: " << point.z << std::endl;
        cloud->push_back(pcl::PointXYZ(point.x,point.y,point.z));
    }
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    viewer->initCameraParameters();

    // Visualize the point cloud
    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }

    return 0;
}