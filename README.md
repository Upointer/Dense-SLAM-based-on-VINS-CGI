# Dense-SLAM based on VINS & CGI
A Dense-SLAM Project for getting Real-time dense 3D map.<br>
## Method
<img src="support_file/Method.png" width = 65% height = 65% div align=center />
<br>
Our approach involves converting the disparity map obtained through stereo matching via CGI-Stereo which is subsequently transformed into a point cloud. Utilizing the localization data derived from stereo images and IMU within the VINS-Fusion framework, we performe positional transformation on the point cloud and then integrate them through a straightforward summation process.<br>

## To Start With
First, you should konw how to deploy a VSLAM project (such as ORB-SLAM3 or VINS-Fusion) on your robot.<br>
Our project is based on [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) and [CGI-Stereo](https://github.com/gangweiX/CGI-Stereo).<br>
Before go to next part, please follow the **Prerequisites** part of VINS-Fusion and make sure your robot can successfully run *Stereo cameras* or *Stereo cameras + IMU*.

### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 20.04.
ROS Noetic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 1.3 **Other Libraries**
* Eigen version: 3.3.4
* Opencv version: 4.2
* PCL verison: 1.8
* [ONNXRUNTIME](http://github.com/microsoft/onnxruntime) version: at least 1.20.0
