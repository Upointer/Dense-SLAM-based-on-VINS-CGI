#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

struct ResizeCropResult {
    Mat outimg;
    double focal_len;
};

void stereoRectification(const Mat& leftImage, const Mat& rightImage,
                         const Mat& leftCameraMatrix, const Mat& leftDistCoeffs,
                         const Mat& rightCameraMatrix, const Mat& rightDistCoeffs,
                         Size imageSize, const Mat& R, const Mat& T,
                         Mat& rectifiedLeft, Mat& rectifiedRight, double& baseline) {
    // distortion correct
    Mat leftUndistorted, rightUndistorted;
    undistort(leftImage, leftUndistorted, leftCameraMatrix, leftDistCoeffs, leftCameraMatrix);
    undistort(rightImage, rightUndistorted, rightCameraMatrix, rightDistCoeffs, rightCameraMatrix);

    // stereo rectify
    Mat R1, R2, P1, P2, Q;
    stereoRectify(leftCameraMatrix, leftDistCoeffs,
                  rightCameraMatrix, rightDistCoeffs,
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 0, imageSize);

    // compute rectify map
    Mat leftMap1, leftMap2, rightMap1, rightMap2;
    initUndistortRectifyMap(leftCameraMatrix, leftDistCoeffs, R1, P1,
                            imageSize, CV_32FC1, leftMap1, leftMap2);
    initUndistortRectifyMap(rightCameraMatrix, rightDistCoeffs, R2, P2,
                            imageSize, CV_32FC1, rightMap1, rightMap2);

    // remap
    remap(leftUndistorted, rectifiedLeft, leftMap1, leftMap2, INTER_LINEAR);
    remap(rightUndistorted, rectifiedRight, rightMap1, rightMap2, INTER_LINEAR);

    // compute baseline
    baseline = norm(T) / 1000;
}

ResizeCropResult resizeCrop(const Mat& inputImage, const Mat& cameraMatrix,
                            int width, int height, double ratio) {
    ResizeCropResult result;

    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    // resize after stereo rectification
    
    Mat resizedImage;
    resize(inputImage, resizedImage, Size(), ratio, ratio, INTER_NEAREST);

    // update params
    fx *= ratio;
    fy *= ratio;
    cx *= ratio;
    cy *= ratio;

    // compute location
    int startX = static_cast<int>(max(cx - width / 2.0, 0.0));
    int startY = static_cast<int>(max(cy - height / 2.0, 0.0));

    startX = min(startX, resizedImage.cols - width);
    startY = min(startY, resizedImage.rows - height);
    startX = max(startX, 0);
    startY = max(startY, 0);

    // crop
    Rect cropArea(startX, startY, width, height);
    Mat croppedImage = resizedImage(cropArea);

    Mat outImage;
    cvtColor(croppedImage, outImage, COLOR_BGR2RGB);
    outImage.convertTo(outImage, CV_32FC3, 1.0 / 255.0);

    result.outimg = outImage;
    result.focal_len = (fx + fy) / 2.0;

    return result;
}

