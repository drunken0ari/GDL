//
// Created by hanta on 24. 4. 30.
//

#include "Util.h"

using namespace cv;
using namespace std;

std::vector<cv::Point2d> Util::Project(const std::vector<cv::Point3d>& points,
                                 const cv::Mat& intrinsic,
                                 const cv::Mat& distortion) {
    std::vector<cv::Point2d> result;
    if (!points.empty()) {
        cv::projectPoints(points, cv::Mat(3, 1, CV_64F, cv::Scalar(0.)),
                          cv::Mat(3, 1, CV_64F, cv::Scalar(0.)), intrinsic,
                          distortion, result);
    }
    return result;
}

vector<Point3f> Util::getCornersInCameraWorld(double side, Vec3d rvec, Vec3d tvec){

    double half_side = side/2;


    // compute rot_mat
    Mat rot_mat;
    Rodrigues(rvec, rot_mat);

    // transpose of rot_mat for easy columns extraction
    Mat rot_mat_t = rot_mat.t();

    // the two E-O and F-O vectors
    double * tmp = rot_mat_t.ptr<double>(0);
    Point3f camWorldE(tmp[0]*half_side,
                      tmp[1]*half_side,
                      tmp[2]*half_side);

    tmp = rot_mat_t.ptr<double>(1);
    Point3f camWorldF(tmp[0]*half_side,
                      tmp[1]*half_side,
                      tmp[2]*half_side);

    // convert tvec to point
    Point3f tvec_3f(tvec[0], tvec[1], tvec[2]);

    // return vector:
    vector<Point3f> ret(4,tvec_3f);

    ret[0] +=  camWorldE + camWorldF;
    ret[1] += -camWorldE + camWorldF;
    ret[2] += -camWorldE - camWorldF;
    ret[3] +=  camWorldE - camWorldF;

    return ret;
}

bool Util::isInImage(int row, int col, int width, int height)
{
    if(row < 0)
        return false;
    if(row >= height)
        return false;
    if(col < 0)
        return false;
    if(col >= width)
        return false;
    return true;
}

std::vector<cv::Point3d> Util::Unproject(const std::vector<cv::Point2d>& points,
                                   const std::vector<double>& Z,
                                   const cv::Mat& intrinsic,
                                   const cv::Mat& distortion){
    double f_x = intrinsic.at<double>(0, 0);
    double f_y = intrinsic.at<double>(1, 1);
    double c_x = intrinsic.at<double>(0, 2);
    double c_y = intrinsic.at<double>(1, 2);
    // This was an error before:
    // double c_x = intrinsic.at<double>(0, 3);
    // double c_y = intrinsic.at<double>(1, 3);

    // Step 1. Undistort
    std::vector<cv::Point2d> points_undistorted;
    assert(Z.size() == 1 || Z.size() == points.size());
    if (!points.empty()) {
        cv::undistortPoints(points, points_undistorted, intrinsic,
                            distortion, cv::noArray(), intrinsic);
    }

    // Step 2. Reproject
    std::vector<cv::Point3d> result;
    result.reserve(points.size());
    for (size_t idx = 0; idx < points_undistorted.size(); ++idx) {
        const double z = Z.size() == 1 ? Z[0] : Z[idx];
        result.push_back(
                cv::Point3d((points_undistorted[idx].x - c_x) / f_x * z,
                            (points_undistorted[idx].y - c_y) / f_y * z, z));
    }
    return result;
}

bool Util::IsParallel(Point3d vec1, Point3d vec2)
{
    if(vec1.x == 0 && vec1.y == 0 && vec1.z == 0)
        return true;
    if(vec2.x == 0 && vec2.y == 0 && vec2.z == 0)
        return true;

    //bool ip = false;
    if(vec1.x == 0 && vec2.x != 0)
        return false;
    if(vec1.y == 0 && vec2.y != 0)
        return false;
    if(vec1.z == 0 && vec2.z != 0)
        return false;

    if(vec1.x == 0 && vec1.y == 0 && vec2.x == 0 && vec2.y == 0)
        return true;
    if(vec1.x == 0 && vec1.z == 0 && vec2.x == 0 && vec2.z == 0)
        return true;
    if(vec1.z == 0 && vec1.y == 0 && vec2.z == 0 && vec2.y == 0)
        return true;

    if(vec1.x == 0 && vec2.x == 0)
        return vec1.y / vec2.y == vec1.z / vec2.z;
    if(vec1.y == 0 && vec2.y == 0)
        return vec1.x / vec2.x == vec1.z / vec2.z;
    if(vec1.z == 0 && vec2.z == 0)
        return vec1.y / vec2.y == vec1.x / vec2.x;

    if(vec2.x == 0)
    {
        return true;
    }
    if(vec2.y == 0)
    {
        return true;
    }
    if(vec2.z == 0)
    {
        return true;
    }
    return (vec1.x / vec2.x == vec1.y / vec2.y) && (vec1.y / vec2.y == vec1.z / vec2.z);
}

Point2i Util::GetGlassPoint(Mat seg)
{
    for (int row = 0; row < seg.rows; row++) {
        for (int col = 0; col < seg.cols; col++) {
            if(seg.at<uchar>(row, col) == 255)
                return Point2i(col, row);
        }
    }
    return Point2i(-1, -1);
}

bool Util::HasSameValue(vector<int> vec)
{
    for(int i = 0; i < vec.size(); i++)
    {
        for(int j = i; j < vec.size(); j++)
        {
            if(i != j)
            {
                if(vec[i] == vec[j])
                {
                    return true;
                }
            }
        }
    }
    return false;
}