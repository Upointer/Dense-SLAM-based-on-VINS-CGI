cmake_minimum_required(VERSION 3.0.2)
project(pcd_spliced)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  pcl_ros
  sensor_msgs
)

## Find PCL
find_package(PCL 1.7 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

## Declare a catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES pcd_spliced
  CATKIN_DEPENDS roscpp pcl_ros sensor_msgs
  DEPENDS PCL
)

## Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

## Add a executable
add_executable(pcd src/pcd.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(pcd
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

catkin_install_python(PROGRAMS
  src/plot_rf2o_odom.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
