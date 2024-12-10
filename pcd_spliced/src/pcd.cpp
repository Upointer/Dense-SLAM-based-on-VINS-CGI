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

#define FRAME "world"
#define DISCUT 3

// init pcl for result
pcl::PointCloud<pcl::PointXYZ>::Ptr total_cloud(new pcl::PointCloud<pcl::PointXYZ>);

class PointCloudMerger {
public:
    PointCloudMerger():frameid(0) {
        // init ros node
        ros::NodeHandle nh;

        // subscriber pointcloud
        pointcloud_subscriber = nh.subscribe<sensor_msgs::PointCloud2>("/cgi_pcl", 10, &PointCloudMerger::pointcloudCallback, this);
        //pointcloud_subscriber = nh.subscribe<sensor_msgs::PointCloud2>("/vins_estimator/point_cloud", 10, &PointCloudMerger::pointcloudCallback, this);

        // publish pointcloud
        pointcloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/merged_pointcloud", 10);

        // init tf listener
        tf_listener = std::make_shared<tf::TransformListener>();
    }

    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_cloud_msg) {
        std::lock_guard<std::mutex> lock(mutex_); // 线程安全
        frameid++;
        //if(frameid % 3 !=1){
        //	return;
        //}

        // tranform
        std::string target_frame = FRAME;
        if (!tf_listener->waitForTransform(target_frame, input_cloud_msg->header.frame_id, input_cloud_msg->header.stamp, ros::Duration(0.1))) {
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
        
        // Create a pass-through filter to keep points within discut meters from frame
        /*
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pass(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(total_cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-DISCUT, DISCUT);
        pass.filter(*cloud_pass);
        pass.setInputCloud(cloud_pass);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-DISCUT, DISCUT);
        pass.filter(*cloud_pass);
        pass.setInputCloud(cloud_pass);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-DISCUT, DISCUT);
        pass.filter(*cloud_pass);
        
        *total_cloud = *cloud_pass;
        */
        // del duplicate points
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(total_cloud);
        vg.setLeafSize(0.1f, 0.1f, 0.1f); // set for proper
        vg.filter(*cloud_filtered);
        
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);
        output.header.frame_id = FRAME; 
        output.header.stamp = ros::Time::now();
        pointcloud_publisher.publish(output);
        std::cout<<"Update"<<std::endl;
    }

private:
    ros::Subscriber pointcloud_subscriber;
    ros::Publisher pointcloud_publisher;
    std::shared_ptr<tf::TransformListener> tf_listener;
    std::mutex mutex_; // 添加互斥锁
    int frameid;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pcd_spliced");
    PointCloudMerger pcm;
    ros::spin();
    return 0;
}
