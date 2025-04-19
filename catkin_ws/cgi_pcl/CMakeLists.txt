cmake_minimum_required(VERSION 3.0.2)
project(cgi_pcl)

set(CMAKE_BUILD_TYPE "Realease")
add_compile_options(-std=c++14)
# find lib
find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
  sensor_msgs
  cv_bridge
  pcl_conversions
  tf
  image_geometry
  rospack
)

find_package(OpenCV 4.2 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_INCLUDE_DIRS "/usr/local/include/onnxruntime")
set(ONNXRUNTIME_LIB_DIR "/usr/local/lib/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_LIB_DIR})

# define catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs
)

# include dir
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# target dir
link_directories(
  ${catkin_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
  ${ONNXRUNTIME_LIB_DIR}
)

# execuate 
add_executable(depth_to_pointcloud src/depth_to_pointcloud.cpp)
target_link_libraries(depth_to_pointcloud
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  pcl_common pcl_io
  ${ONNXRUNTIME_LIB}
)

add_executable(depth_estimator src/depth_estimator.cpp)
target_link_libraries(depth_estimator
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  pcl_common
  ${ONNXRUNTIME_LIB}
  tinyxml2
  ${CUDA_LIBRARIES}
)
