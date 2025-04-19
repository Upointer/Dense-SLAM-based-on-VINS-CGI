#include <ros/ros.h>
#include <ros/package.h>
#include <tinyxml2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <image_geometry/pinhole_camera_model.h>

class DepthToPointCloud {
public:
    DepthToPointCloud() : nh_("~") {
        // 参数配置
        nh_.param<std::string>("depth_topic", depth_topic_, "/cgi/depth");
        nh_.param<std::string>("cloud_topic", cloud_topic_, "/cgi/pointcloud");
        // nh_.param<std::string>("camera_info", camera_info_, "/cgi/camera_info");
        nh_.param("queue_size", queue_size_, 5);

        // 初始化订阅和发布
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(cloud_topic_, queue_size_);
        // 从XML文件加载相机参数
        std::string package_path = ros::package::getPath("cgi_pcl");
        std::string config_path = package_path + "/config/camera_params.xml";
        loadCameraParamsFromXML(config_path);
        depth_sub_ = nh_.subscribe(depth_topic_, queue_size_, &DepthToPointCloud::depthCallback, this);
    }

    // void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& info) {
    //     cam_model_.fromCameraInfo(info);
    // }

    void depthCallback(const sensor_msgs::ImageConstPtr& depth_msg) {
        if (cam_model_.fx() == 0 || cam_model_.fy() == 0) {
            ROS_WARN("Camera model not initialized");
            return;
        }

        try {
            // 转换深度图像
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(depth_msg);
            
            // 创建点云
            pcl::PointCloud<pcl::PointXYZ> cloud;
            cloud.header.frame_id = depth_msg->header.frame_id;
            cloud.width = cv_ptr->image.cols;
            cloud.height = cv_ptr->image.rows;
            cloud.is_dense = false;
            cloud.points.resize(cloud.width * cloud.height);

            // 坐标转换
            for (int v = 0; v < cv_ptr->image.rows; ++v) {
                for (int u = 0; u < cv_ptr->image.cols; ++u) {
                    float depth = cv_ptr->image.at<float>(v, u);
                    if (std::isnan(depth) || depth >= 50.0 || depth <= 0.0) continue;

                    // 使用image_geometry进行坐标转换
                    // 转换到ROS右手坐标系（X右，Y下，Z前） 
                    cv::Point2d uv(u, v);
                    cv::Point3d ray = cam_model_.projectPixelTo3dRay(uv);
                    
                    // 调整坐标系：X右，Y左，Z上（右手系）
                    //cv::Point3d xyz = ray * depth;
                    cv::Point3d xyz;
                    xyz.x = (u - cam_model_.cx())*depth/cam_model_.fx();
                    xyz.y = (v - cam_model_.cy())*depth/cam_model_.fy();
                    xyz.z = depth;
                    //xyz.y = -xyz.y;  // 翻转Y轴
                    //xyz.z = -xyz.z;  // 翻转Z轴

                    // 过滤无效点
                    if (!std::isfinite(xyz.x) || !std::isfinite(xyz.y) || !std::isfinite(xyz.z)) {
                        continue;
                    }

                    pcl::PointXYZ point;
                    point.x = xyz.x;
                    point.y = xyz.y;  
                    point.z = xyz.z;
                    cloud.at(u, v) = point;
                }
            }

            // 发布点云
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(cloud, cloud_msg);
            cloud_msg.header = depth_msg->header;
            cloud_msg.header.frame_id = "camera";
            cloud_pub_.publish(cloud_msg);
            std::cout<<"Point cloud published"<<std::endl;
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

private:
    void loadCameraParamsFromXML(const std::string& file_path) {
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(file_path.c_str()) != tinyxml2::XML_SUCCESS) {
            ROS_ERROR("Failed to load camera parameters file: %s", file_path.c_str());
            return;
        }

        // 修改XML节点查找逻辑
        auto root = doc.FirstChildElement("opencv_storage");
        if (!root) {
            ROS_ERROR("Missing opencv_storage root element");
            return;
        }

        // 读取左相机内参
        auto left_cam = root->FirstChildElement("left_cameraMatrix");
        auto left_data = left_cam->FirstChildElement("data")->GetText();
        std::stringstream left_stream(left_data);
        double fx, fy, cx, cy, dummy;
        left_stream >> fx >> dummy >> cx >> dummy >> fy >> cy >> dummy >> dummy >> dummy;
        fx *= ratio; fy *= ratio; cx *= ratio; cy *= ratio;
        //ROS_INFO("camera params: %.2f,%.2f,%.2f,%.2f",fx,fy,cx,cy);
        
        // 读取平移向量计算基线
        auto tvec = root->FirstChildElement("translation_vector");
        auto tdata = tvec->FirstChildElement("data")->GetText();
        std::stringstream t_stream(tdata);
        double tx, ty, tz;
        t_stream >> tx >> ty >> tz;
        float baseline = std::abs(tx);  // 基线为平移向量的x分量绝对值

        image_geometry::PinholeCameraModel model;
        sensor_msgs::CameraInfo camera_info;
        camera_info.width = 640;  // 根据实际图像尺寸设置
        camera_info.height = 480;
        camera_info.K = { 
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0 
        };
        camera_info.P = {
            fx, 0.0, cx, -fx*baseline,  // 添加基线到投影矩阵
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        };
        model.fromCameraInfo(camera_info);
        cam_model_ = model;
        ROS_INFO("Camera parameters loaded successfully");
    }

    ros::NodeHandle nh_;
    ros::Subscriber depth_sub_;
    // ros::Subscriber cam_info_sub_;
    ros::Publisher cloud_pub_;
    image_geometry::PinholeCameraModel cam_model_;
    
    std::string depth_topic_;
    std::string cloud_topic_;
    // std::string camera_info_;
    int queue_size_;
    double ratio = 0.2625;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "depth_to_pointcloud");
    DepthToPointCloud converter;
    ros::spin();
    return 0;
}
