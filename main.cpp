#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "Util.h"
#include "GPDL.h"

using namespace cv;
using namespace std;

const double f_x = 320.0;
const double f_y = 320.0;
const double c_x = 320.0;
const double c_y = 240.0;

const int windowSize = 4;
const unsigned short distThreshold = 1500;
const int numOfrp = 3;
const int maxIter = 100;
const double dstInlierRate1 = 0.65;
const double dstInlierRate2 = 0.15;
const double inlierThreshold = 0.1;//meter

int main() {

    //random_device rd;
    //mt19937 mt(rd());
    srand(time(NULL));


    const cv::Mat intrinsic =
            (cv::Mat_<double>(3, 3) << f_x, 0.0, c_x, 0.0, f_y, c_y, 0.0, 0.0, 1.0);
    const cv::Mat distortion =
            (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0);  // This works!


    //string pathDepth = "/home/hanta/hanta_data_com_for_RFE/EcoBuilding";
    string pathDepth = "/home/hanta/hanta_real_agr_infer2/depth";
    //string pathSeg = "/home/hanta/RFENet/results2/test/rgb";
    string pathSeg = "/home/hanta/RFENet/agr2/test/rgb";
    string pathList = "/home/hanta/hanta_real_agr_infer2/times";
    string pathAruco = "/home/hanta/hanta_real_agr_infer2/aruco";

    //ifstream file(pathDepth + ".txt");
    ifstream file(pathList + ".txt");
    string content;

    vector<double> distances;
    vector<double> angles;
    vector<double> arucoDis;

    while(getline(file, content)) {
        //content = "0000169";
        //Mat imgDepth = imread(pathDepth + "/depth_left_planar/" + content + ".png", IMREAD_ANYDEPTH);
        //Mat imgSegOri = imread(pathSeg + "/" + content + "_color.png", IMREAD_COLOR);

        Mat imgDepth = imread(pathDepth + "/" + content + ".png", IMREAD_ANYDEPTH);
        int imgWidth = imgDepth.cols;
        int imgHeight = imgDepth.rows;
        string content2 = content;
        Mat imgSegOri = imread(pathSeg + "/" + content2.replace(content2.size() - 7, 0, "color") + "_color.png",
                               IMREAD_COLOR);

        Mat imgSegDst;
        Mat imgSeg;
        Mat separatedSeg;


        resize(imgSegOri, imgSegDst, Size(imgWidth, imgHeight), 0, 0, INTER_NEAREST);

        imgSeg = Mat(imgHeight, imgWidth, CV_8UC1, Scalar(0));

        Mat imgEdge = Mat(imgHeight, imgWidth, CV_8UC3, Scalar(0, 0, 0));
        Mat imgEdgeCddt = Mat(imgHeight, imgWidth, CV_8UC3, Scalar(0, 0, 0));

        for (int row = 0; row < imgHeight; row++) {
            for (int col = 0; col < imgWidth; col++) {
                if (imgSegDst.at<Vec3b>(row, col) != Vec3b(0, 0, 0)) {
                    imgSeg.at<uchar>(row, col) = 255;
                }
            }
        }

        separatedSeg = Mat(imgHeight, imgWidth, CV_8UC1, Scalar(0));

        GPDL::SeparateBinaryImageSegments(imgSeg, separatedSeg);

        GPDL::MakeEdgeImage(imgSeg, separatedSeg, imgEdge);

        GPDL::ExtractCandidatesDepthAware(imgDepth, imgEdge, imgEdgeCddt, distThreshold, windowSize);


        imshow("imgDepth", imgDepth);
        imshow("imgSeg", imgSeg);
        imshow("imgEdge", imgEdge);
        imshow("imgEdgeFlag", imgEdgeCddt);
        imshow("separatedSeg", separatedSeg);
        waitKey(1);


        vector<int> segments;

        for (int row = 0; row < imgHeight; row++) {
            for (int col = 0; col < imgWidth; col++) {
                int v = (int) imgEdgeCddt.at<Vec3b>(row, col)[0];
                if (v != 0) {
                    auto it = find(segments.begin(), segments.end(), v);
                    if (it == segments.end()) {
                        segments.push_back(v);
                    }
                }
            }
        }

        vector<double> segDistances;
        vector<double> segAngles;
        vector<double> segAruco;

        for (int s = 0; s < segments.size(); s++) {
            const int windowSize = 5;

            vector<Point3d> glassEdges;
            vector<bool> isInlier;
            Point3d outerProductRef;
            double rhRef;

            bool isSuccess = GPDL::EstimatePlaneBySegment(windowSize, imgEdgeCddt, segments[s], imgDepth, &glassEdges,
                                                          &isInlier, intrinsic, distortion, numOfrp, inlierThreshold,
                                                          dstInlierRate1, dstInlierRate2, maxIter, &outerProductRef,
                                                          &rhRef);

            vector<Point3d> trueEdges;
            if (isSuccess) {

                for (int x = 0; x < glassEdges.size(); x++) {
                    if (isInlier[x]) {
                        trueEdges.push_back(glassEdges[x]);
                    }
                }

                vector<Point2d> uvs = Util::Project(trueEdges, intrinsic, distortion);

                string textfile = pathAruco + "/" + content + ".png.txt";

                ifstream openFile(textfile);
                if (openFile.is_open()) {
                    string line;
                    getline(openFile, line);
                    double tx = stod(line);
                    getline(openFile, line);
                    double ty = stod(line);
                    getline(openFile, line);
                    double tz = stod(line);
                    getline(openFile, line);
                    double rx = stod(line);
                    getline(openFile, line);
                    double ry = stod(line);
                    getline(openFile, line);
                    double rz = stod(line);
                    openFile.close();

                    vector<Point3f> arucoCorners = Util::getCornersInCameraWorld(0.176, Vec3d(rx, ry, rz),
                                                                                 Vec3d(tx, ty, tz));

                    Point3f vec1 = arucoCorners[1] - arucoCorners[0];
                    Point3f vec2 = arucoCorners[2] - arucoCorners[0];

                    Point3d outerProduct = Point3d(vec1.y * vec2.z - vec1.z * vec2.y,
                                                   vec1.z * vec2.x - vec1.x * vec2.z,
                                                   vec1.x * vec2.y - vec1.y * vec2.x);

                    double rh =
                            outerProduct.x * arucoCorners[0].x + outerProduct.y * arucoCorners[0].y +
                            outerProduct.z * arucoCorners[0].z;

                    segAruco.push_back(
                            sqrt(arucoCorners[0].x * arucoCorners[0].x + arucoCorners[0].y * arucoCorners[0].y +
                                 arucoCorners[0].z * arucoCorners[0].z));

                    Point3d arucoNormal = Point3d(outerProduct.x, outerProduct.y, outerProduct.z);

                    double cosTheta = abs(arucoNormal.x * outerProductRef.x + arucoNormal.y * outerProductRef.y +
                                          arucoNormal.z * outerProductRef.z)
                                      / (sqrt(arucoNormal.x * arucoNormal.x + arucoNormal.y * arucoNormal.y +
                                              arucoNormal.z * arucoNormal.z)
                                         * sqrt(outerProductRef.x * outerProductRef.x +
                                                outerProductRef.y * outerProductRef.y +
                                                outerProductRef.z * outerProductRef.z));

                    double thetaAngle = acos(cosTheta);
                    segAngles.push_back(thetaAngle);

                    double sumOfDis = 0.0;
                    for (int tr = 0; tr < trueEdges.size(); tr++) {
                        sumOfDis += abs(arucoNormal.x * trueEdges[tr].x + arucoNormal.y * trueEdges[tr].y +
                                        arucoNormal.z * trueEdges[tr].z - rh)
                                    / sqrt(arucoNormal.x * arucoNormal.x + arucoNormal.y * arucoNormal.y +
                                           arucoNormal.z * arucoNormal.z);
                    }
                    double avgOfDis = sumOfDis / (double) trueEdges.size();

                    segDistances.push_back(avgOfDis);
                }

            }
        }

    }
}
