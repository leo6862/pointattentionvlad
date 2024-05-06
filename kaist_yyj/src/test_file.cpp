#include <iostream>
#include <filesystem>

std::string extractFilename(const std::string& path)
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


int main()
{
    std::string folderPath = "/home/yyj/download/kaist/urban17/sensor_data/VLP_left";  // 指定文件夹路径

    // std::string folderPath = "path/to/folder";  // 指定文件夹路径

    for (const auto& entry : std::filesystem::directory_iterator(folderPath))
    { 
        std::cout << extractFilename(entry.path()) << std::endl; // 打印的是abs path
    }
    return 0;
}