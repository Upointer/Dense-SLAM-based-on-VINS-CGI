#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <queue>
#include <mutex>
#include <tinyxml2.h>
#include <image_process.h>
#include <ros/package.h>

class DepthEstimator {
public:
    DepthEstimator() : nh_("~"), env_(ORT_LOGGING_LEVEL_WARNING) {
        // params init
        std::string model_path = "models/cgi_stereo_sceneflow_480x640.onnx";
        std::string left_topic_ = "/1/pylon_camera_node/image_raw";
        std::string right_topic_ = "/3/pylon_camera_node/image_raw";
        std::string depth_topic_ = "/cgi/depth";
        queue_size_ = 5;

        // load params from XML
        std::string xml_path = ros::package::getPath("cgi_pcl") + "/config/camera_params.xml";
        loadCameraParamsFromXML(xml_path);

        // init ONNX model
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

        // init pub & sub
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

        // read XML
        auto root = doc.FirstChildElement("opencv_storage");
        if (!root) {
            ROS_ERROR("Invalid XML structure: missing opencv_storage");
            return;
        }

        // left camera params
        parseMatrix(root->FirstChildElement("left_cameraMatrix"), left_camera_matrix_);
        parseMatrix(root->FirstChildElement("left_distCoeffs"), left_dist_coeffs_);
        
        // right camera params
        parseMatrix(root->FirstChildElement("right_cameraMatrix"), right_camera_matrix_);
        parseMatrix(root->FirstChildElement("right_distCoeffs"), right_dist_coeffs_);

        // rotation and translation
        parseMatrix(root->FirstChildElement("rotation_matrix"), R_);
        parseMatrix(root->FirstChildElement("translation_vector"), T_);

        // trans m to mm
        T_ *= 1000.0;
    }

    void parseMatrix(tinyxml2::XMLElement* elem, cv::Mat& output) {
        if (!elem) {
            ROS_ERROR("Missing matrix element");
            return;
        }

        // matrix dimension
        int rows = elem->FirstChildElement("rows")->IntText();
        int cols = elem->FirstChildElement("cols")->IntText();
        std::string data_str = elem->FirstChildElement("data")->GetText();

        // trans data to cv::Mat
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
            // trans msg to cv::Mat
            cv::Mat left_img = cv_bridge::toCvShare(left_msg, "bgr8")->image;
            cv::Mat right_img = cv_bridge::toCvShare(right_msg, "bgr8")->image;

            // stereo rectification
            cv::Mat rect_left, rect_right;
            double baseline;
            stereoRectification(left_img, right_img,
                              left_camera_matrix_, left_dist_coeffs_,
                              right_camera_matrix_, right_dist_coeffs_,
                              cv::Size(2448, 2048), R_, T_,
                              rect_left, rect_right, baseline);


            // resize and crop
            auto res_left = resizeCrop(rect_left, left_camera_matrix_, 640, 480, 0.2625);
            auto res_right = resizeCrop(rect_right, right_camera_matrix_, 640, 480, 0.2625);        

            // prepare tensor
            std::vector<float> left_tensor = prepareInput(res_left.outimg);
            std::vector<float> right_tensor = prepareInput(res_right.outimg);

            // reasoning
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

            // process output
            float* disp_data = outputs[0].GetTensorMutableData<float>();
            cv::Mat disp_map(480, 640, CV_32FC1, disp_data);

            // compute depth
            //const double focal = (res_left.focal_len + res_right.focal_len) / 2.0;
            //cv::Mat depth_map = (baseline * focal) / (disp_map + 1e-9);
            const double focal_ratio = res_left.focal_len / res_right.focal_len;
            cv::Mat depth_map = (baseline * res_left.focal_len) / (disp_map * focal_ratio + 1e-9);
            
            // pub depth
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
        
        // channel separation & normlization
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
        valid_depth.setTo(0, depth < 0);  // filter negative depth

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "32FC1", valid_depth).toImageMsg();
        depth_pub_.publish(msg);
        ROS_DEBUG("Published depth map");
    }


    // camera params
    cv::Mat left_camera_matrix_;
    cv::Mat left_dist_coeffs_;
    cv::Mat right_camera_matrix_;
    cv::Mat right_dist_coeffs_;
    cv::Mat R_;
    cv::Mat T_;

    // ROS utils
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

    // queue size
    int queue_size_;

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "depth_estimator");
    DepthEstimator estimator;
    ros::spin();
    return 0;
}
