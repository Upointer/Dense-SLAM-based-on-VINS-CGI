#!/usr/bin/env python3

import onnxruntime as ort
import numpy as np
import cv2
import time
import rospy
import tf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import queue

model_path="/home/xjtu/catkin_ws/src/ros_pointcloud/scripts/cgi_stereo_sceneflow_480x640.onnx"

baseline=0.505
f_l = 351
f_r = 360
downsample_factor=2
near_distance = 2.0
cut_distance = 40.0 #Distance Filter Parament

#left cam paraments
left_cameraMatrix = np.array([[703.2074 ,   0.     , 596.6635],
           [0.     , 702.3131, 492.0989],
           [0.     ,   0.     ,   1.     ]])
left_distCoeffs = np.array([0.0006777292124847927, -0.005108393841913188, -4.087354716903757e-05, 0.002199175887778906])
#right cam paraments
right_cameraMatrix = np.array([[720.8368,   0.     , 601.6054],
           [0.     , 719.8317, 488.4360],
           [0.     ,   0.     ,   1.     ]])
right_distCoeffs = np.array([0.007810194984131187, -0.00858821855402433, -0.00010617177737143683, 0.000878167665866135])
# Rotation
R = np.array([[0.9995492389463141, 0.004348636476727187, 0.029705357810292862],
  [-0.00481896241935607, 0.999863877917023, 0.015779836435566075],
  [-0.029632693482791662, -0.015915872502808716, 0.9994341341376275]])
# Translation
T = np.array([-504.4725995730913, 3.147710295289657, 48.66907821003241])
#image_size
image_size = (1224, 1024)
#Global Value
left_image_queue = queue.Queue()
right_image_queue = queue.Queue()
MAX_QUEUE_LENGTH = 10
PROCESS_EVERY_N_FRAMES = 1
frame_counter = 0


available_providers = ort.get_available_providers()
#print(f"Available providers: {available_providers}")
providers = []
if "CUDAExecutionProvider" in available_providers:
    providers.append("CUDAExecutionProvider")
    print("GPU is working")
else:
    print("GPU is not working")
    providers.append("CPUExecutionProvider")

# 初始化 ONNX 模型
sess = ort.InferenceSession(
    model_path,
    providers=providers
)

def image_callback(msg, args):
    bridge, image_type = args
    global left_image_queue, right_image_queue, frame_counter

    # SlowDown Input Frequence
    frame_counter += 1
    if frame_counter % PROCESS_EVERY_N_FRAMES != 0:
        return

    # 将 ROS 图像消息转换为 OpenCV 图像
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    # Save Images & timestamp for left or right
    if image_type == "left":
        left_image = Lens_Distortion_Correction(cv_image, left_cameraMatrix, left_distCoeffs)
        left_image_time = msg.header.stamp.to_sec()
        if left_image_queue.qsize() >= MAX_QUEUE_LENGTH:
            left_image_queue.get()
        left_image_queue.put((left_image_time,left_image))
    elif image_type == "right":
        right_image = Lens_Distortion_Correction(cv_image, right_cameraMatrix ,right_distCoeffs)
        right_image_time = msg.header.stamp.to_sec()
        if right_image_queue.qsize() >= MAX_QUEUE_LENGTH:
            right_image_queue.get()
        right_image_queue.put((right_image_time,right_image))

    #Synchronization
    sync_images()

def sync_images():
    global left_image_queue, right_image_queue

    while not left_image_queue.empty() and not right_image_queue.empty():
        left_time, left_image = left_image_queue.queue[0]
        right_time, right_image = right_image_queue.queue[0]

        time_diff = abs(left_time - right_time)
        if time_diff <= 0.05:
            left_image_queue.get()
            right_image_queue.get()
            process_images(left_image, right_image, sess)
        elif left_time < right_time:
            left_image_queue.get()
        else:
            right_image_queue.get()


def process_images(left_image, right_image, sess):        

    input_names = [i.name for i in sess.get_inputs()]
    output_names = [i.name for i in sess.get_outputs()]

    input_height = sess.get_inputs()[0].shape[2]
    input_width = sess.get_inputs()[0].shape[3]
    #print(f"{input_height=}")
    #print(f"{input_width=}")
    
    left_image,right_image = stereo_rectification(left_image,right_image,left_cameraMatrix,left_distCoeffs,right_cameraMatrix,right_distCoeffs,image_size,R,T)

    left = Crop(left_image, input_width, input_height)
    right = Crop(right_image, input_width, input_height)

    left = np.transpose(left, (2, 0, 1))[np.newaxis, :, :, :]
    right = np.transpose(right, (2, 0, 1))[np.newaxis, :, :, :]

    
    t = time.time()
    output = sess.run(output_names,
        {
            input_names[0]: left,
            input_names[1]: right,
        }
    )
    dt = time.time() - t
    print(f"\033[34mElapsed: {dt:.3f} sec, {1/dt:.3f} FPS\033[0m")
    disp = output[0][0]
    
    # Convert disparity to depth    
    focal_ratio = f_l / f_r
    disp_adjusted = disp * focal_ratio
    depth = (baseline * f_l) / (disp_adjusted + 1e-9)
    
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
    cv2.imwrite(f"o.png", colored_depth)
    # 降采样深度图
    downsampled_height = input_height // downsample_factor
    downsampled_width = input_width // downsample_factor
    depth_downsampled = cv2.resize(depth, (downsampled_width, downsampled_height), interpolation=cv2.INTER_AREA)
    colored_depth_downsampled = cv2.resize(colored_depth, (downsampled_width, downsampled_height), interpolation=cv2.INTER_AREA)
    
    # 深度转点云
    height, width = depth_downsampled.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    points = np.zeros((height, width, 3), dtype=np.float32)
    points[:, :, 0] = (xx - width / 2) * depth_downsampled / (f_l / downsample_factor)
    points[:, :, 1] = (yy - height / 2) * depth_downsampled / (f_l / downsample_factor)
    points[:, :, 2] = depth_downsampled

    # 发布点云
    publish_pointcloud(points, colored_depth_downsampled)

# Undistort
def Lens_Distortion_Correction(image, cameraMatrix, distCoeffs):
    new_img = cv2.undistort(image, cameraMatrix, distCoeffs, None, cameraMatrix)
    return new_img

def stereo_rectification(left_image,right_image,left_cameraMatrix,left_distCoeffes,right_cameraMatrix,right_distCoeffes,image_size,R,T):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_cameraMatrix, left_distCoeffes, right_cameraMatrix, right_distCoeffes, image_size, R, T, alpha=0
    )
    map1_x, map1_y = cv2.initUndistortRectifyMap(left_cameraMatrix, left_distCoeffes, R1, P1, image_size, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(right_cameraMatrix, right_distCoeffes, R2, P2, image_size, cv2.CV_32FC1)
    rectified_left = cv2.remap(left_image, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map2_x, map2_y, cv2.INTER_LINEAR)
    return rectified_left,rectified_right

# Crop
def Crop(img, width, height):
    print(img.shape[1],img.shape[0])
    print(width,height)
    img = cv2.resize(img,(0,0),fx=0.525,fy=0.525)
    # 获取图像的中心点
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    print(center_x, center_y)

    # 计算裁减区域的左上角坐标
    start_x = int(max(center_x - width/2 , 0))
    start_y = int(max(center_y - height/2 , 0))
    print(start_x, start_y)
    # 确保裁减区域不超出图像边界
    end_x = int(start_x + width)
    end_y = int(start_y + height)
    print(end_x, end_y)
    # 裁减图像
    cropped = img[start_y:end_y, start_x:end_x]
    print(cropped.shape[1], cropped.shape[0])
    outimg = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), (width, height)).astype(np.float32) / 255.0
    #cv2.imwrite(f"2.png", outimg)

    return outimg


def publish_pointcloud(points, colored_depth):
    pub = rospy.Publisher('/cgi_pcl', PointCloud2, queue_size=10)
    
    # TF listener
    tf_listener = tf.TransformListener()
    
    # Wait for available transform
    try:
        tf_listener.waitForTransform('world','camera',rospy.Time(0),rospy.Duration(1.0))
        (trans, rot) = tf_listener.lookupTransform('world','camera',rospy.Time(0))
        transform_mat = tf_listener.fromTranslationRotation(trans, rot)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logwarn("TF transform not available")
        return

    # 创建PointCloud2消息
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.UINT8, 1),
        PointField('g', 13, PointField.UINT8, 1),
        PointField('b', 14, PointField.UINT8, 1),
    ]

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'world'

    # 将点云数据和颜色信息组合
    point_list = []
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            x, y, z = points[i, j]
            r, g, b = colored_depth[i, j]
            #point_list.append([x, y, z, int(r), int(g), int(b)])
            if near_distance*near_distance <= x*x+y*y+z*z <= cut_distance*cut_distance:
                point_world = np.dot(transform_mat, np.array([x,y,z,1.0]))[:3]
                point_list.append([point_world[0], point_world[1], point_world[2], int(r), int(g), int(b)])

    pc2_data = pc2.create_cloud(header, fields, point_list)

    # 发布PointCloud2消息
    pub.publish(pc2_data)
    print(f"Published point cloud data to topic: {pub.name}") 
    

if __name__ == "__main__":
    rospy.init_node('depth_estimation_node', anonymous=True)
    bridge = CvBridge()
    print("Thread Start")
    left_sub = rospy.Subscriber("/left/pylon_camera_node/image_raw", Image, image_callback, callback_args=(bridge, "left"))
    right_sub = rospy.Subscriber("/right/pylon_camera_node/image_raw", Image, image_callback, callback_args=(bridge, "right"))

    rospy.spin()
