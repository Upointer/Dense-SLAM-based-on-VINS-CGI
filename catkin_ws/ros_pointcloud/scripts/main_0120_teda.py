#!/usr/bin/env python3

import cv2
import time
import rospy
import tf
import queue
import std_msgs.msg
import numpy as np
import onnxruntime as ort
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField
from Image_process import stereo_rectification, resize_crop

# Resolution and resize ratio
image_size = (2448,2048)
ratio = 0.2625
# Left camera paraments
left_cameraMatrix = np.array([[1195.6998107014192 ,   0.     , 1205.323641673825],
           [0.     , 1194.9064694677256, 989.8900109329693],
           [0.     ,   0.     ,   1.     ]])
left_distCoeffs = np.array([-0.06281511564879048, 0.07205839047009635, 0.00037437660934028325, -0.00029026594014092377])
# Right camera paraments
right_cameraMatrix = np.array([[1198.5137849127186,   0.     , 1189.3664305969146],
           [0.     , 1197.6145792437346, 978.7713115133586],
           [0.     ,   0.     ,   1.     ]])
right_distCoeffs = np.array([-0.0214097613949015, 0.024978433170190387, 0.00018235609651712377, -0.00030136313785876306])
# Stereo paraments
## Rotation
R = np.array([[0.9996741857473295, -0.0009672658622501354, -0.025506602030780055],
  [0.0011695987504849296, 0.9999679614598689, 0.007918844142170848],
  [0.0254981252088802, -0.007946096559749006, 0.9996432889587598]])
## Translation (mm)
T = np.array([-876.7712875596468, -2.0200417425657504, -6.922912031092199])

#Global Value
model_path="/home/xjtu/catkin_ws/src/ros_pointcloud/scripts/cgi_stereo_sceneflow_480x640.onnx" #/path/to/your/model
left_image_queue = queue.Queue()
right_image_queue = queue.Queue()
MAX_QUEUE_LENGTH = 10
PROCESS_EVERY_N_FRAMES = 1
frame_counter = 0
downsample_factor=1
#Distance Filter Parament
near_distance = 2.0
cut_distance = 50.0 

#####----------main body----------#####

available_providers = ort.get_available_providers()

providers = []
if "CUDAExecutionProvider" in available_providers:
    providers.append("CUDAExecutionProvider")
    print("GPU is working")
else:
    print("GPU is not working")
    providers.append("CPUExecutionProvider")

# init ONNX model
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

    # ROS image message to OpenCV image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    # Save Images & timestamp for left or right
    if image_type == "left":
        left_image = cv_image
        #left_image_time = msg.header.stamp.to_sec()
        left_image_time = msg.header.stamp
        if left_image_queue.qsize() >= MAX_QUEUE_LENGTH:
            left_image_queue.get()
        left_image_queue.put((left_image_time,left_image))
    elif image_type == "right":
        right_image = cv_image
        #right_image_time = msg.header.stamp.to_sec()
        right_image_time = msg.header.stamp
        if right_image_queue.qsize() >= MAX_QUEUE_LENGTH:
            right_image_queue.get()
        right_image_queue.put((right_image_time,right_image))

    #Synchronization
    sync_images()

def sync_images():
    global left_image_queue, right_image_queue,timestamp 

    while not left_image_queue.empty() and not right_image_queue.empty():
        left_time, left_image = left_image_queue.queue[0]
        right_time, right_image = right_image_queue.queue[0]
        
        left_time_1=left_time.to_sec()
        right_time_1=right_time.to_sec()
        
        time_diff = abs(left_time_1 - right_time_1)
        if time_diff <= 0.05:
            left_image_queue.get()
            right_image_queue.get()
            timestamp=left_time
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
    cv2.imwrite("rl.png",left_image)
    left_image,right_image, baseline = stereo_rectification(left_image,right_image,left_cameraMatrix,left_distCoeffs,right_cameraMatrix,right_distCoeffs,image_size,R,T)
    print(baseline)
    cv2.imwrite("sl.png",left_image)
    left, f_l = resize_crop(left_image, left_cameraMatrix, input_width, input_height, ratio)
    right, f_r = resize_crop(right_image, right_cameraMatrix, input_width, input_height, ratio)
    l_norm = ((left - left.min()) / (left.max() - left.min()) * 255).astype(np.uint8)
    r_norm = ((right - right.min()) / (right.max() - right.min()) * 255).astype(np.uint8)
    cv2.imwrite("l.png",l_norm)
    cv2.imwrite("r.png",r_norm)
    #print(f_l, f_r)
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
    cv2.imwrite("output.png",disp)
    # Convert disparity to depth    
    focal_ratio = f_l / f_r
    disp_adjusted = disp * focal_ratio
    depth = (baseline * f_l) / (disp_adjusted + 1e-9)
    
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    #cv2.imwrite("oo.png",depth_norm)
    colored_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
    cv2.imwrite(f"o.png", colored_depth)
    # downsample depth image
    downsampled_height = input_height // downsample_factor
    downsampled_width = input_width // downsample_factor
    depth_downsampled = cv2.resize(depth, (downsampled_width, downsampled_height), interpolation=cv2.INTER_AREA)
    colored_depth_downsampled = cv2.resize(colored_depth, (downsampled_width, downsampled_height), interpolation=cv2.INTER_AREA)
    
    # depth to pointcloud
    height, width = depth_downsampled.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    points = np.zeros((height, width, 3), dtype=np.float32)
    points[:, :, 0] = (xx - width / 2) * depth_downsampled / (f_l / downsample_factor)
    points[:, :, 1] = (yy - height / 2) * depth_downsampled / (f_l / downsample_factor)
    points[:, :, 2] = depth_downsampled

    # pulish
    publish_pointcloud(points, colored_depth_downsampled)


def publish_pointcloud(points, colored_depth):
    pub = rospy.Publisher('/cgi_pcl', PointCloud2, queue_size=10)
    
    # TF listener
    tf_listener = tf.TransformListener()
    
    # Wait for available transform
    try:
        tf_listener.waitForTransform('body','camera',rospy.Time(0),rospy.Duration(1.0))
        (trans, rot) = tf_listener.lookupTransform('body','camera',rospy.Time(0))
        transform_mat = tf_listener.fromTranslationRotation(trans, rot)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logwarn("TF transform not available")
        return

    # generate PointCloud2 message
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.UINT8, 1),
        PointField('g', 13, PointField.UINT8, 1),
        PointField('b', 14, PointField.UINT8, 1),
    ]

    header = std_msgs.msg.Header()
    header.stamp = timestamp
    header.frame_id = 'body'

    # plus rgb to pcl

    height, width, _ = points.shape
    homogeneous_points = np.ones((height * width, 4))
    homogeneous_points[:, :3] = points.reshape(-1, 3)

    world_homogeneous = np.dot(homogeneous_points, transform_mat.T)
    world_points = world_homogeneous[:, :3]  # get XYZ

    colors = colored_depth.reshape(-1, 3)

    dist_sq = np.sum(homogeneous_points**2, axis=1) 

    mask = (dist_sq >= near_distance**2) & (dist_sq <= cut_distance**2)

    filtered_points = world_points[mask]
    filtered_colors = colors[mask]
    
    combined = np.empty(filtered_points.shape[0], dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('r', np.uint8), ('g', np.uint8), ('b', np.uint8)
    ])
    
    combined['x'] = filtered_points[:, 0]
    combined['y'] = filtered_points[:, 1]
    combined['z'] = filtered_points[:, 2]
    combined['r'] = filtered_colors[:, 0]
    combined['g'] = filtered_colors[:, 1]
    combined['b'] = filtered_colors[:, 2]

    pc2_data = pc2.create_cloud(header, fields, combined)

    # publish PointCloud2 message
    pub.publish(pc2_data)
    print(f"Published point cloud data to topic: {pub.name}") 
    

if __name__ == "__main__":
    rospy.init_node('depth_estimation_node', anonymous=True)
    bridge = CvBridge()
    print("Thread Start")
    left_sub = rospy.Subscriber("/1/pylon_camera_node/image_raw", Image, image_callback, callback_args=(bridge, "left"))
    right_sub = rospy.Subscriber("/3/pylon_camera_node/image_raw", Image, image_callback, callback_args=(bridge, "right"))

    rospy.spin()


