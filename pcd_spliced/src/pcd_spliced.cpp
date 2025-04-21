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
#include <deque>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>

#define WORLDFRAME "world"
#define BODYFRAME "body"
#define MAX_QUEUE_SIZE 80
#define XYDISCUT 30.0
#define ZDISCUT 5.0

class PointCloudMerger {
public:
    PointCloudMerger() {
        // init ros node
        ros::NodeHandle nh("~");
	
        // pointcloud subscriber
        pointcloud_subscriber = nh.subscribe<sensor_msgs::PointCloud2>("/cgi/pointcloud", 10, &PointCloudMerger::pointcloudCallback, this);
        
        // pointcloud publisher
        pointcloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/merged_pointcloud", 10);
        
        // init tf listener
        tf_listener = std::make_shared<tf::TransformListener>();        
    }
   
    
    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_cloud_msg) {
        
        std::string world_frame = WORLDFRAME;
        std::string body_frame = BODYFRAME;
        
        // Get the transform from input point frame to world frame
        tf::StampedTransform transform;
        try {
            tf_listener->lookupTransform(world_frame, input_cloud_msg->header.frame_id, input_cloud_msg->header.stamp, transform);
        }
        catch (tf::TransformException &ex) {
            try {
                tf_listener->waitForTransform(world_frame, input_cloud_msg->header.frame_id, input_cloud_msg->header.stamp,ros::Duration(0.1));
                tf_listener->lookupTransform(world_frame, input_cloud_msg->header.frame_id, input_cloud_msg->header.stamp, transform);
            }
            catch (tf::TransformException &ex_wait) {
                tf_listener->lookupTransform(world_frame, input_cloud_msg->header.frame_id, ros::Time(0), transform);
            }
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input_cloud_msg, *cloud);      

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(0.05f, 0.05f, 0.05f); // Grid filter
        vg.filter(*cloud_filtered);
        
	// transform to world frame before push to the queue
        pcl::PointCloud<pcl::PointXYZ>::Ptr world_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl_ros::transformPointCloud(*cloud_filtered, *world_cloud, transform);
	
	// queue check and pop
	if (cloud_queue.size() >= MAX_QUEUE_SIZE){
	    cloud_queue.pop_front();
	    //ROS_INFO("PointCloud QUEUE POP!");
	}
	
        // push to queue end
        cloud_queue.push_back(world_cloud);
        
        // merge all cloud in queue
        pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& queued_cloud : cloud_queue) {     
            // Add the transformed cloud to the merged cloud
            *merged_cloud += *queued_cloud;
        }      
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> tvg;
        tvg.setInputCloud(merged_cloud);
        tvg.setLeafSize(0.1f, 0.1f, 0.3f);
        tvg.filter(*merged_cloud_filtered);
        
        // Get the transform from world frame to body frame 
        tf_listener->lookupTransform(body_frame, world_frame, ros::Time(0), transform);
        
        // Tranform merged cloud to body frame
        pcl::PointCloud<pcl::PointXYZ>::Ptr bodyframe_merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl_ros::transformPointCloud(*merged_cloud_filtered, *bodyframe_merged_cloud, transform);
        
        // Create a pass-through filter to keep points within discut meters from bodyframe
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pass(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(bodyframe_merged_cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-XYDISCUT, XYDISCUT);
        pass.filter(*cloud_pass);
        pass.setInputCloud(cloud_pass);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-XYDISCUT, XYDISCUT);
        pass.filter(*cloud_pass);
        pass.setInputCloud(cloud_pass);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-ZDISCUT, ZDISCUT);
        pass.filter(*cloud_pass);
        
        //Get the transform from body frame to world frame
        tf_listener->lookupTransform(world_frame, body_frame, ros::Time(0), transform);
        
        // Tranform merged cloud back to world frame
        pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud_cut(new pcl::PointCloud<pcl::PointXYZ>);
        pcl_ros::transformPointCloud(*cloud_pass, *merged_cloud_cut, transform);
        

        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*merged_cloud_cut, output);
        output.header.frame_id = WORLDFRAME; 
        output.header.stamp = ros::Time::now();
        pointcloud_publisher.publish(output);
        ROS_INFO("Merged point cloud Updated");
    }

private:
    ros::Subscriber pointcloud_subscriber;
    ros::Publisher pointcloud_publisher;
    std::shared_ptr<tf::TransformListener> tf_listener;
    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_queue;
    double radius, height, resolution;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pcd_spliced");
    PointCloudMerger pcm;
    ros::spin();
    return 0;
}

