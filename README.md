# Dense-SLAM based on VINS & CGI
A Dense-SLAM Project for building Real-time dense 3D map.<br>
## 1. Method
<img src="support_file/Method.png" width = 65% height = 65% div align=center />
<br>
Our approach involves converting the disparity map obtained through stereo matching via CGI-Stereo which is subsequently transformed into a point cloud. Utilizing the localization data derived from stereo images and IMU within the VINS-Fusion framework, we performe positional transformation on the point cloud and then integrate them through a straightforward summation process.<br>

### 1.1. Depth Image & Point Cloud
In package **ros_pointcloud** we fused this tow parts together, while in package **cgi_pcl** we use two nodes: **depth_estimator** and **depth_to_point** to generate depth image and point cloud separately.

### 1.2. Point Cloud Merge
This package subscribe tf tree produced by VINS-Fusion and point cloud topic, add them together after processed by voxel filter, distance filter and time filter.

## 2. To Start With
First, you should konw how to deploy a VSLAM project (such as ORB-SLAM3 or VINS-Fusion) on your robot.<br>
Our project is based on [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) and [CGI-Stereo](https://github.com/gangweiX/CGI-Stereo).<br>
Before go to the next part, please follow the **Prerequisites** part of VINS-Fusion and make sure your robot can successfully run *Stereo cameras* or *Stereo cameras + IMU*.

### 2.1. **Ubuntu** and **ROS**
Ubuntu 64-bit 20.04.
ROS Noetic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 2.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 2.3. **Other Libraries**
* **Eigen** version: 3.3.4
* **Opencv** version: 4.2
* **PCL** verison: 1.8
* [**ONNX Runtime**](http://github.com/microsoft/onnxruntime) version: 1.20.0
* **CUDA** (optional but recommended) if you want to use GPU for inference

## 3. Build DSVC
### 3.1. Clone and Make
Clone the repository and catkin_make:
```
    cd ~/catkin_ws/src
    git clone https://github.com/Upointer/Dense-SLAM-based-on-VINS-CGI.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```
(It's recommended to put VINS-Fusion under the same workspace.)
### 3.2. Get **ONNX** inference for **CGI_Stereo**
We provide two ways for you to onnx inference models.
* You can follow [this repository](https://github.com/fateshelled/cgistereo_demo) or find **358_CGI-Stereo** in [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) to trans into ONNX framework.
* Our demo will provide you a readymade model to run our demo rosbag.
## 4. Demo Example
Download [Our demo ROS bag](https://pan.baidu.com/s/11w9-92u1pqxjAzpFOfljbA?pwd=ts6h) to YOUR_ROSBAG_FOLDER then copy the config file to config folder of VINS-Fusion and copy the models directory to **cgi_pcl** or **ros_pointcloud**.
```
   cd ~/YOUR_ROSBAG_FOLDER/demo_rosbag
   cp -r demo_car/ ~/catkin_ws/src/VINS-Fusion/config
   cp -r models/ ~/catkin_ws/src/cgi_pcl
   cp -r models/ ~/catkin_ws/src/ros_pointcloud
```
### 4.1. Run Demo
We provide **cgi_pcl** ros package based on C++ and **ros_pointcloud** ros package based on python3.10, We will show our demo on **cgi_pcl**. 
Open five terminals, run vins odometry, point cloud and point cloud merge, rviz and play the bag file.
```
    roslaunch vins vins_rviz.launch
    rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion/config/demo_car/demo_car.yaml 
    roslaunch cgi_pcl cgi_pcl.launch
    roslaunch pcd_spliced pcd_spliced.launch
    rosbag play YOUR_DATASET_FOLDER/demo_car.bag
```
### 4.2. Run your own project
If you want to run this repository on your own robot, thera are two things you should care about.
* **Camera Params** Calibrate your stereo camera, and write your camera params to **cammera_params.xml** in config.
* **Resolution and Crop Ratio** As the input size of inference models is fixed, for different resolution, you should choose a certain ratio to resize your raw images and crop as less as possible.

