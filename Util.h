//
// Created by hanta on 24. 4. 30.
//

#ifndef GLASSPLANEDETECTIONANDLOCALIZATION_UTIL_H
#define GLASSPLANEDETECTIONANDLOCALIZATION_UTIL_H

#include <vector>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

class Util {
public:
    static std::vector<cv::Point2d> Project(const std::vector<cv::Point3d>& points,
                                           const cv::Mat& intrinsic,
                                           const cv::Mat& distortion);

    static vector<Point3f> getCornersInCameraWorld(double side, Vec3d rvec, Vec3d tvec);

    static bool isInImage(int row, int col, int width, int height);

    static std::vector<cv::Point3d> Unproject(const std::vector<cv::Point2d>& points,
                                             const std::vector<double>& Z,
                                             const cv::Mat& intrinsic,
                                             const cv::Mat& distortion);

    static bool IsParallel(Point3d vec1, Point3d vec2);

    static Point2i GetGlassPoint(Mat seg);

    static bool HasSameValue(vector<int> vec);
};



#endif //GLASSPLANEDETECTIONANDLOCALIZATION_UTIL_H
