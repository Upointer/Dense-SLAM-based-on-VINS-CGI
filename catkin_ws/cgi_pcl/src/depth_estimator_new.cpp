#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <queue>
#include <mutex>
#include <cmath>
#include <tinyxml2.h>
#include <ros/package.h>

class DepthEstimator {
public:
    struct ResizeCropResult {
        cv::Mat outimg;
        double focal_len;
    };
    DepthEstimator() : nh_("~"), env_(ORT_LOGGING_LEVEL_WARNING) {
        // 参数初始化
        std::string model_path = "/home/xjtu/catkin_ws/src/cgi_pcl/models/cgi_stereo_sceneflow_480x640.onnx";
        std::string left_topic_ = "/1/pylon_camera_node/image_raw";
        std::string right_topic_ = "/3/pylon_camera_node/image_raw";
        std::string depth_topic_ = "/cgi/depth";
        queue_size_ = 5;

        // 从XML加载相机参数
        std::string xml_path = ros::package::getPath("cgi_pcl") + "/config/camera_params.xml";
        loadCameraParamsFromXML(xml_path);

        // 初始化ONNX模型
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        try{
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options); //启用CUDA
            ROS_INFO("CUDA provider enabled");
        } catch (const Ort::Exception& e) {
            ROS_WARN("Failed to enable CUDA: %s", e.what());
        }
            
        session_ = Ort::Session(env_, model_path.c_str(), session_options);

        // 初始化发布订阅
        depth_pub_ = nh_.advertise<sensor_msgs::Image>(depth_topic_, queue_size_);
        left_sub_ = nh_.subscribe(left_topic_, queue_size_, &DepthEstimator::leftImageCallback, this);
        right_sub_ = nh_.subscribe(right_topic_, queue_size_, &DepthEstimator::rightImageCallback, this);
    }

private:
    void loadCameraParamsFromXML(const std::string& file_path) {
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(file_path.c_str()) != tinyxml2::XML_SUCCESS) {
            ROS_ERROR("Failed to load camera parameters file: %s", file_path.c_str());
            return;
        }

        // 解析XML结构
        auto root = doc.FirstChildElement("opencv_storage");
        if (!root) {
            ROS_ERROR("Invalid XML structure: missing opencv_storage");
            return;
        }

        // 读取左相机内参
        parseMatrix(root->FirstChildElement("left_cameraMatrix"), left_camera_matrix_);
        parseMatrix(root->FirstChildElement("left_distCoeffs"), left_dist_coeffs_);
        
        // 读取右相机内参
        parseMatrix(root->FirstChildElement("right_cameraMatrix"), right_camera_matrix_);
        parseMatrix(root->FirstChildElement("right_distCoeffs"), right_dist_coeffs_);

        // 读取旋转和平移矩阵
        parseMatrix(root->FirstChildElement("rotation_matrix"), R_);
        parseMatrix(root->FirstChildElement("translation_vector"), T_);

        // 转换平移向量单位（XML中单位为米，转换为毫米）
        T_ *= 1000.0;
    }

    void parseMatrix(tinyxml2::XMLElement* elem, cv::Mat& output) {
        if (!elem) {
            ROS_ERROR("Missing matrix element");
            return;
        }

        // 解析矩阵维度
        int rows = elem->FirstChildElement("rows")->IntText();
        int cols = elem->FirstChildElement("cols")->IntText();
        std::string data_str = elem->FirstChildElement("data")->GetText();

        // 转换数据到OpenCV矩阵
        std::stringstream ss(data_str);
        output = cv::Mat(rows, cols, CV_64F);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double val;
                ss >> val;
                output.at<double>(i, j) = val;
            }
        }
    }

    void leftImageCallback(const sensor_msgs::ImageConstPtr& msg) {
        processImage(msg, left_queue_);
    }

    void rightImageCallback(const sensor_msgs::ImageConstPtr& msg) {
        processImage(msg, right_queue_);
    }

    void processImage(const sensor_msgs::ImageConstPtr& msg, 
                     std::queue<sensor_msgs::ImageConstPtr>& queue) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue.size() >= queue_size_) queue.pop();
        queue.push(msg);
        synchronizeImages();
    }

    void synchronizeImages() {
        while (!left_queue_.empty() && !right_queue_.empty()) {
            auto left = left_queue_.front();
            auto right = right_queue_.front();

            const double time_diff = std::abs((left->header.stamp - right->header.stamp).toSec());
            if (time_diff <= 0.03) {
                processStereoPair(left, right);
                left_queue_.pop();
                right_queue_.pop();
            }
            else if (left->header.stamp < right->header.stamp) {
                left_queue_.pop();
            }
            else {
                right_queue_.pop();
            }
        }
    }

    void processStereoPair(const sensor_msgs::ImageConstPtr& left_msg,
                          const sensor_msgs::ImageConstPtr& right_msg) {
        try {
            // 转换为OpenCV图像
            cv::Mat left_img = cv_bridge::toCvShare(left_msg, "bgr8")->image;
            cv::Mat right_img = cv_bridge::toCvShare(right_msg, "bgr8")->image;

            //cv::imwrite("/home/xjtu/catkin_ws/src/cgi_pcl/src/rl.png", left_img);

            // 立体校正
            cv::Mat rect_left, rect_right;
            double baseline;
            stereoRectification(left_img, right_img,
                              left_camera_matrix_, left_dist_coeffs_,
                              right_camera_matrix_, right_dist_coeffs_,
                              cv::Size(2448, 2048), R_, T_,
                              rect_left, rect_right, baseline);

            cv::imwrite("/home/xjtu/catkin_ws/src/cgi_pcl/src/sl.png", rect_left);

            // 图像预处理
            auto res_left = resizeCrop(rect_left, left_camera_matrix_, 640, 480, 0.2625);
            auto res_right = resizeCrop(rect_right, right_camera_matrix_, 640, 480, 0.2625);
            
            cv::Mat t;
            cv::normalize(res_left.outimg, t, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite("/home/xjtu/catkin_ws/src/cgi_pcl/src/l.png", t);
            cv::normalize(res_right.outimg, t, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite("/home/xjtu/catkin_ws/src/cgi_pcl/src/r.png", t);

            // 准备输入张量
            std::vector<float> left_tensor = prepareInput(res_left.outimg);
            std::vector<float> right_tensor = prepareInput(res_right.outimg);

            // 运行推理
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
            
            std::vector<Ort::Value> input_tensors;
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, left_tensor.data(), left_tensor.size(), input_shape_.data(), 4));
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, right_tensor.data(), right_tensor.size(), input_shape_.data(), 4));

            auto outputs = session_.Run(Ort::RunOptions{nullptr}, 
                                      input_names_.data(), input_tensors.data(), 2,
                                      output_names_.data(), 1);

            // 处理输出
            float* disp_data = outputs[0].GetTensorMutableData<float>();
            cv::Mat disp_map(480, 640, CV_32FC1, disp_data);

            cv::Mat disp_vis;
            cv::normalize(disp_map, disp_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite("/home/xjtu/catkin_ws/src/cgi_pcl/src/o.png", disp_vis);

            // 计算深度图
            //const double focal = (res_left.focal_len + res_right.focal_len) / 2.0;
            //cv::Mat depth_map = (baseline * focal) / (disp_map + 1e-9);
            const double focal_ratio = res_left.focal_len / res_right.focal_len;
            cv::Mat depth_map = (baseline * res_left.focal_len) / (disp_map * focal_ratio + 1e-9);
            
            // 发布深度图
            publishDepth(depth_map, left_msg->header);
        }
        catch (const cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    std::vector<float> prepareInput(const cv::Mat& image) {
        cv::Mat float_img;
        image.convertTo(float_img, CV_32FC3);

        std::vector<float> input_tensor;
        input_tensor.reserve(3 * 480 * 640);
        
        // 通道分离与归一化
        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels);
        for (const auto& channel : channels) {
            const float* p = channel.ptr<float>();
            input_tensor.insert(input_tensor.end(), p, p + channel.total());
        }
        return input_tensor;
    }

    void publishDepth(const cv::Mat& depth, const std_msgs::Header& header) {
        cv::Mat valid_depth = depth.clone();
        valid_depth.setTo(0, depth < 0);  // 过滤负深度值

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "32FC1", valid_depth).toImageMsg();
        depth_pub_.publish(msg);
        ROS_DEBUG("Published depth map");
    }
    
    void stereoRectification(const cv::Mat& leftImage, const cv::Mat& rightImage,
                         const cv::Mat& leftCameraMatrix, const cv::Mat& leftDistCoeffs,
                         const cv::Mat& rightCameraMatrix, const cv::Mat& rightDistCoeffs,
                         cv::Size imageSize, const cv::Mat& R, const cv::Mat& T,
                         cv::Mat& rectifiedLeft, cv::Mat& rectifiedRight, double& baseline) {
        // distortion correct
        cv::Mat leftUndistorted, rightUndistorted;
        cv::undistort(leftImage, leftUndistorted, leftCameraMatrix, leftDistCoeffs, leftCameraMatrix);
        cv::undistort(rightImage, rightUndistorted, rightCameraMatrix, rightDistCoeffs, rightCameraMatrix);

        // stereo rectify
        cv::Mat R1, R2, P1, P2, Q;
        cv::stereoRectify(leftCameraMatrix, leftDistCoeffs,
                  rightCameraMatrix, rightDistCoeffs,
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  0, 0, imageSize);
    
        std::cout<<CV_VERSION<<std::endl;
        std::cout<<"R1: "<<R1<<"R2: "<<R2<<"P1: "<<P1<<"P2: "<<P2;

        // compute rectify map
        cv::Mat leftMap1, leftMap2, rightMap1, rightMap2;
        cv::initUndistortRectifyMap(leftCameraMatrix, leftDistCoeffs, R1, P1,
                            imageSize, CV_32FC1, leftMap1, leftMap2);
        cv::initUndistortRectifyMap(rightCameraMatrix, rightDistCoeffs, R2, P2,
                            imageSize, CV_32FC1, rightMap1, rightMap2);

        // remap
        cv::remap(leftUndistorted, rectifiedLeft, leftMap1, leftMap2, cv::INTER_LINEAR);
        cv::remap(rightUndistorted, rectifiedRight, rightMap1, rightMap2, cv::INTER_LINEAR);

        // compute baseline
        baseline = norm(T) / 1000;
    }

    ResizeCropResult resizeCrop(const cv::Mat& inputImage, const cv::Mat& cameraMatrix,
                            int width, int height, double ratio) {
        ResizeCropResult result;

        double fx = cameraMatrix.at<double>(0, 0);
        double fy = cameraMatrix.at<double>(1, 1);
        double cx = cameraMatrix.at<double>(0, 2);
        double cy = cameraMatrix.at<double>(1, 2);

        // resize after stereo rectification
    
        cv::Mat resizedImage;
        cv::resize(inputImage, resizedImage, cv::Size(), ratio, ratio, cv::INTER_NEAREST);
        cv::imwrite("/home/xjtu/catkin_ws/src/cgi_pcl/src/re.png", resizedImage);

        // update params
        fx *= ratio;
        fy *= ratio;
        cx *= ratio;
        cy *= ratio;

        // compute location
        int startX = static_cast<int>(std::max(cx - width / 2.0, 0.0));
        int startY = static_cast<int>(std::max(cy - height / 2.0, 0.0));

        startX = std::min(startX, resizedImage.cols - width);
        startY = std::min(startY, resizedImage.rows - height);
        startX = std::max(startX, 0);
        startY = std::max(startY, 0);

        // crop
        cv::Rect cropArea(startX, startY, width, height);
        cv::Mat croppedImage = resizedImage(cropArea);

        cv::Mat outImage;
        cv::cvtColor(croppedImage, outImage, cv::COLOR_BGR2RGB);
        outImage.convertTo(outImage, CV_32FC3, 1.0 / 255.0);

        result.outimg = outImage;
        result.focal_len = (fx + fy) / 2.0;

        return result;
    }

    // 相机参数（现从XML加载）
    cv::Mat left_camera_matrix_;
    cv::Mat left_dist_coeffs_;
    cv::Mat right_camera_matrix_;
    cv::Mat right_dist_coeffs_;
    cv::Mat R_;
    cv::Mat T_;

    // ROS组件
    ros::NodeHandle nh_;
    ros::Subscriber left_sub_, right_sub_;
    ros::Publisher depth_pub_;
    std::queue<sensor_msgs::ImageConstPtr> left_queue_, right_queue_;
    std::mutex mutex_;

    // ONNX Runtime
    Ort::Env env_;
    Ort::Session session_{nullptr};
    std::vector<const char*> input_names_ = {"left", "right"};
    std::vector<const char*> output_names_ = {"output"};
    std::vector<int64_t> input_shape_ = {1, 3, 480, 640};
    
    int queue_size_;

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "depth_estimator");
    DepthEstimator estimator;
    ros::spin();
    return 0;
}
