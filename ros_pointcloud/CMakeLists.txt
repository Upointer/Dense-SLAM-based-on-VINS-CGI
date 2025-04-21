cmake_minimum_required(VERSION 3.0.2)
project(ros_pointcloud)

# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  sensor_msgs
  std_msgs
)

# find_package(Boost REQUIRED COMPONENTS system)

# catkin_python_setup()

# add_message_files(

# add_service_files(

# add_action_files(

# generate_messages(

# generate_dynamic_reconfigure_options(

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

# add_library(${PROJECT_NAME}

# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


# add_executable(${PROJECT_NAME}_node src/ros_pointcloud_node.cpp)

# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# target_link_libraries(${PROJECT_NAME}_node

# catkin_install_python(PROGRAMS

# install(TARGETS ${PROJECT_NAME}_node

# install(TARGETS ${PROJECT_NAME}

# install(DIRECTORY include/${PROJECT_NAME}/


# install(FILES

# catkin_add_gtest(${PROJECT_NAME}-test test/test_ros_pointcloud.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

# catkin_add_nosetests(test)
