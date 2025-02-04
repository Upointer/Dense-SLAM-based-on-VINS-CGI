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
#define MAX_QUEUE_SIZE 45
#define XYDISCUT 15.0
#define ZDISCUT 5.0

class PointCloudMerger {
public:
    PointCloudMerger() {
        // init ros node
        ros::NodeHandle nh("~");

        // get paraments from ROS server
        if (!nh.getParam("disk_radius", radius)) {
            ROS_WARN("Parameter 'disk_radius' not found, using default value: 1.0");
            radius = 1.0;
        }
        if (!nh.getParam("disk_height", height)) {
            ROS_WARN("Parameter 'disk_height' not found, using default value: 0.4");
            height = 0.4;
        }
        if (!nh.getParam("disk_resolution", resolution)) {
            ROS_WARN("Parameter 'disk_resolution' not found, using default value: 0.05");
            resolution = 0.05;
        }
        ROS_INFO("Disk parameters loaded: radius=%.2f, height=%.2f, resolution=%.2f", radius, height, resolution);
	
	// init disk pointcloud
	generateInitialDiskCloud();
	
        // pointcloud subscriber
        pointcloud_subscriber = nh.subscribe<sensor_msgs::PointCloud2>("/cgi_pcl", 10, &PointCloudMerger::pointcloudCallback, this);
        
        // pointcloud publisher
        pointcloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/merged_pointcloud", 10);
        
        // init tf listener
        tf_listener = std::make_shared<tf::TransformListener>();        
    }
    
    void generateInitialDiskCloud() {
        pcl::PointCloud<pcl::PointXYZ>::Ptr disk_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (double r = 0; r < radius; r += resolution) {
            int points_on_ring = static_cast<int>(2 * M_PI * r / resolution);
            for (int i = 0; i < points_on_ring; ++i) {
                double angle = i * 2 * M_PI / points_on_ring;
                pcl::PointXYZ point;
                point.x = r * cos(angle);
                point.y = r * sin(angle);
                point.z = height;
                disk_cloud->points.push_back(point);
            }
        }
        
        // push disk pointcloud to queue
        cloud_queue.push_back(disk_cloud);
        ROS_INFO("Initial Disk cloud generated ");
    }
    
    

    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_cloud_msg) {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // sensor_msgs::PointCloud2 tranform to pcl::PointCloud<pcl::PointXYZ>
        pcl::fromROSMsg(*input_cloud_msg, *cloud);      

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(0.08f, 0.08f, 0.08f); // Grid filter
        vg.filter(*cloud_filtered);
	
	// queue check and pop
	if (cloud_queue.size() >= MAX_QUEUE_SIZE){
	    cloud_queue.pop_front();
	    ROS_INFO("PointCloud QUEUE POP!");
	}
	
        // push to queue end
        cloud_queue.push_back(cloud_filtered);
        
        // merge all cloud in queue
        pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& queued_cloud : cloud_queue) {     
            // Add the transformed cloud to the merged cloud
            *merged_cloud += *queued_cloud;
        }      
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> tvg;
        tvg.setInputCloud(merged_cloud);
        tvg.setLeafSize(0.08f, 0.08f, 10.0f); // filter for axis Z
        tvg.filter(*merged_cloud_filtered);
        
        std::string world_frame = WORLDFRAME;
        std::string body_frame = BODYFRAME;
        // Get the transform from world frame to body frame 
        tf::StampedTransform transform;
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

