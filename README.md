# Dense-SLAM based on VINS & CGI
A Dense-SLAM Project for getting Real-time dense 3D map.<br>
## 1. Method
<img src="support_file/Method.png" width = 65% height = 65% div align=center />
<br>
Our approach involves converting the disparity map obtained through stereo matching via CGI-Stereo which is subsequently transformed into a point cloud. Utilizing the localization data derived from stereo images and IMU within the VINS-Fusion framework, we performe positional transformation on the point cloud and then integrate them through a straightforward summation process.<br>

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
<br>

## 4. Demo Example
Download [Our demo ROS bag]() to YOUR_ROSBAG_FOLDER and copy the config file to config folder of VINS-Fusion.
```
   cd ~/YOUR_ROSBAG_FOLDER/demo_rosbag
   cp -r demo_car/ ~/catkin_ws/src/VINS-Fusion/config  
```

### 4.1. Run Demo
