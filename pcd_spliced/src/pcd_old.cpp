#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>

// init pcl for result
pcl::PointCloud<pcl::PointXYZ>::Ptr total_cloud(new pcl::PointCloud<pcl::PointXYZ>);

class PointCloudMerger {
public:
    PointCloudMerger() {
        // init ros node
        ros::NodeHandle nh;

        // subscriber pointcloud
        pointcloud_subscriber = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/points", 10, &PointCloudMerger::pointcloudCallback, this);

        // publish pointcloud
        pointcloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/merged_pointcloud", 10);

        // init tf listener
        tf_listener = std::make_shared<tf::TransformListener>();
    }

    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_cloud_msg) {
        std::lock_guard<std::mutex> lock(mutex_); // 线程安全

        // tranform
        std::string target_frame = "odom";
        if (!tf_listener->waitForTransform(target_frame, input_cloud_msg->header.frame_id, input_cloud_msg->header.stamp, ros::Duration(5.0))) {
            ROS_WARN("Waiting for transform timed out.");
            return;
        }
        sensor_msgs::PointCloud2 transformed_cloud;
        pcl_ros::transformPointCloud(target_frame, *input_cloud_msg, transformed_cloud, *tf_listener);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // sensor_msgs::PointCloud2 tranform to pcl::PointCloud<pcl::PointXYZ>
        pcl::fromROSMsg(transformed_cloud, *cloud);      

        // plus to total cloud
        *total_cloud += *cloud;
        
        // Create a pass-through filter to keep points within 2.6 meters from odom
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pass(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(total_cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-1.5, 1.5);
        pass.filter(*cloud_pass);
        pass.setInputCloud(cloud_pass);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-1.5, 1.5);
        pass.filter(*cloud_pass);
        pass.setInputCloud(cloud_pass);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-1.5, 1.5);
        pass.filter(*cloud_pass);
        
        *total_cloud = *cloud_pass;
        
        // del duplicate points
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud_pass);
        vg.setLeafSize(0.02f, 0.02f, 0.02f); // set for proper
        vg.filter(*cloud_filtered);
        
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);
        output.header.frame_id = "odom"; 
        output.header.stamp = ros::Time::now();
        pointcloud_publisher.publish(output);
    }

private:
    ros::Subscriber pointcloud_subscriber;
    ros::Publisher pointcloud_publisher;
    std::shared_ptr<tf::TransformListener> tf_listener;
    std::mutex mutex_; // 添加互斥锁
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pcd_spliced");
    PointCloudMerger pcm;
    ros::spin();
    return 0;
}
