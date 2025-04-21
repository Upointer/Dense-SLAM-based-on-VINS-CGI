# Dense-SLAM based on VINS & CGI
A Dense-SLAM Project for getting Real-time dense 3D map.<br>
## To Start With
First, you should konw how to deploy a VSLAM project (such as ORB-SLAM3 or VINS-Fusion) on your robot.<br>
Our project is based on [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) and [CGI-Stereo](https://github.com/gangweiX/CGI-Stereo).<br>
Before go to next part, please follow the **Prerequisites** part of VINS-Fusion and make sure your robot can successfully run *Stereo cameras* or *Stereo cameras + IMU*.
## Method
<img src="support_file/Method.png" width = 65% height = 65% div align=center />
Our approach involves converting the disparity map obtained through stereo matching via CGI-Stereo which is subsequently transformed into a point cloud. Utilizing the localization data derived from stereo images and IMU within the VINS-Fusion framework, we performe positional transformation on the point cloud and then integrate them through a straightforward summation process. 
