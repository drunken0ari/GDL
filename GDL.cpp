//
// Created by hanta on 24. 4. 30.
//

#include "GPDL.h"

void GPDL::SeparateBinaryImageSegments(Mat imgSeg, Mat& separatedSeg)
{
    int numOfSeg;
    numOfSeg = 128;

    vector<Point2i> pointStack;
    Mat segVisited;
    imgSeg.copyTo(segVisited);
    while (Util::GetGlassPoint(segVisited) != Point2i(-1, -1)) {
        pointStack.push_back(Util::GetGlassPoint(segVisited));
        do {
            Point2i pt2i = pointStack[pointStack.size() - 1];
            pointStack.pop_back();
            segVisited.at<uchar>(pt2i.y, pt2i.x) = 0;
            separatedSeg.at<uchar>(pt2i.y, pt2i.x) = numOfSeg;

            if (Util::isInImage(pt2i.y - 1, pt2i.x, segVisited.cols, segVisited.rows)) {
                if (segVisited.at<uchar>(pt2i.y - 1, pt2i.x) != 0) {
                    pointStack.push_back(Point2i(pt2i.x, pt2i.y - 1));
                }
            }
            if (Util::isInImage(pt2i.y + 1, pt2i.x, segVisited.cols, segVisited.rows)) {
                if (segVisited.at<uchar>(pt2i.y + 1, pt2i.x) != 0) {
                    pointStack.push_back(Point2i(pt2i.x, pt2i.y + 1));
                }
            }
            if (Util::isInImage(pt2i.y, pt2i.x - 1, segVisited.cols, segVisited.rows)) {
                if (segVisited.at<uchar>(pt2i.y, pt2i.x - 1) != 0) {
                    pointStack.push_back(Point2i(pt2i.x - 1, pt2i.y));
                }
            }
            if (Util::isInImage(pt2i.y, pt2i.x + 1, segVisited.cols, segVisited.rows)) {
                if (segVisited.at<uchar>(pt2i.y, pt2i.x + 1) != 0) {
                    pointStack.push_back(Point2i(pt2i.x + 1, pt2i.y));
                }
            }
        } while (pointStack.size() != 0);
        numOfSeg++;
    }
}

void GPDL::MakeEdgeImage(Mat imgSeg, Mat separatedSeg, Mat& imgEdge)
{
    for (int row = 0; row < imgSeg.rows; row++) {
        for (int col = 0; col < imgSeg.cols; col++) {
            if (imgSeg.at<uchar>(row, col) != 0) {
                if (Util::isInImage(row - 1, col, imgSeg.cols, imgSeg.rows)) {
                    if (imgSeg.at<uchar>(row - 1, col) == 0) {
                        imgEdge.at<Vec3b>(row, col) = Vec3b(separatedSeg.at<uchar>(row, col), separatedSeg.at<uchar>(row, col),
                                                            separatedSeg.at<uchar>(row, col));
                    }
                }
                if (Util::isInImage(row + 1, col, imgSeg.cols, imgSeg.rows)) {
                    if (imgSeg.at<uchar>(row + 1, col) == 0) {
                        imgEdge.at<Vec3b>(row, col) = Vec3b(separatedSeg.at<uchar>(row, col), separatedSeg.at<uchar>(row, col),
                                                            separatedSeg.at<uchar>(row, col));
                    }
                }
                if (Util::isInImage(row, col - 1, imgSeg.cols, imgSeg.rows)) {
                    if (imgSeg.at<uchar>(row, col - 1) == 0) {
                        imgEdge.at<Vec3b>(row, col) = Vec3b(separatedSeg.at<uchar>(row, col), separatedSeg.at<uchar>(row, col),
                                                            separatedSeg.at<uchar>(row, col));
                    }
                }
                if (Util::isInImage(row, col + 1, imgSeg.cols, imgSeg.rows)) {
                    if (imgSeg.at<uchar>(row, col + 1) == 0) {
                        imgEdge.at<Vec3b>(row, col) = Vec3b(separatedSeg.at<uchar>(row, col), separatedSeg.at<uchar>(row, col),
                                                            separatedSeg.at<uchar>(row, col));
                    }
                }
            }
        }
    }
}

void GPDL::ExtractCandidatesDepthAware(Mat imgDepth, Mat imgEdge, Mat& imgEdgeCddt, double distThreshold, int windowSize)
{
    for (int row = 0; row < imgDepth.rows; row++) {
        for (int col = 0; col < imgDepth.cols; col++) {
            if (imgEdge.at<Vec3b>(row, col) != Vec3b(0, 0, 0)) {
                for (int i = -windowSize; i < windowSize; i++) {
                    for (int j = -windowSize; j < windowSize; j++) {
                        int irow = row + i;
                        int jcol = col + j;
                        if (Util::isInImage(irow, jcol, imgEdgeCddt.cols, imgEdgeCddt.rows)) {
                            if (imgEdgeCddt.at<Vec3b>(irow, jcol) != Vec3b(255, 255, 255)) {
                                if (Util::isInImage(irow - 1, jcol, imgDepth.cols, imgDepth.rows)) {
                                    if (abs(imgDepth.at<unsigned short>(irow, jcol) -
                                            imgDepth.at<unsigned short>(irow - 1, jcol)) > distThreshold) {
                                        imgEdgeCddt.at<Vec3b>(irow, jcol) = imgEdge.at<Vec3b>(row, col);
                                    }
                                }
                                if (Util::isInImage(irow + 1, jcol, imgDepth.cols, imgDepth.rows)) {
                                    if (abs(imgDepth.at<unsigned short>(irow, jcol) -
                                            imgDepth.at<unsigned short>(irow + 1, jcol)) > distThreshold) {
                                        imgEdgeCddt.at<Vec3b>(irow, jcol) = imgEdge.at<Vec3b>(row, col);
                                    }
                                }
                                if (Util::isInImage(irow, jcol - 1, imgDepth.cols, imgDepth.rows)) {
                                    if (abs(imgDepth.at<unsigned short>(irow, jcol) -
                                            imgDepth.at<unsigned short>(irow, jcol - 1)) > distThreshold) {
                                        imgEdgeCddt.at<Vec3b>(irow, jcol) = imgEdge.at<Vec3b>(row, col);
                                    }
                                }
                                if (Util::isInImage(irow, jcol + 1, imgDepth.cols, imgDepth.rows)) {
                                    if (abs(imgDepth.at<unsigned short>(irow, jcol) -
                                            imgDepth.at<unsigned short>(irow, jcol + 1)) > distThreshold) {
                                        imgEdgeCddt.at<Vec3b>(irow, jcol) = imgEdge.at<Vec3b>(row, col);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

bool GPDL::EstimatePlaneBySegment(const int windowSize, Mat imgEdgeCddt, int segmentNum, Mat imgDepth
                                  , vector<Point3d>* glassEdges, vector<bool>* isInlier
                                  , const Mat intrinsic, const Mat distortion, int numOfRp
                                  , double inlierThreshold, double dstInlierRate1, double dstInlierRate2, int maxIter
                                  , Point3d* outerProductRef, double* rhRef)
{
    vector<Point2d> uvValues;
    vector<double> depths;
    vector<bool> isLocalMin;

    for (int row = 0; row < imgEdgeCddt.rows; row++) {
        for (int col = 0; col < imgEdgeCddt.cols; col++) {
            if (imgEdgeCddt.at<Vec3b>(row, col) == Vec3b(segmentNum, segmentNum, segmentNum)) {
                double depth = (double) (imgDepth.at<unsigned short>(row, col)) / 1000.0;
                Point2d inputPt;
                inputPt.x = (double) col;
                inputPt.y = (double) row;
                depths.push_back(depth);
                uvValues.push_back(inputPt);

                int ir = row / windowSize * windowSize;
                int ic = col / windowSize * windowSize;
                bool isMin = true;

                for (int iir = 0; iir < windowSize; iir++) {
                    for (int iic = 0; iic < windowSize; iic++) {
                        if (imgEdgeCddt.at<Vec3b>(ir + iir, ic + iic) ==
                            Vec3b(segmentNum, segmentNum, segmentNum)) {
                            if (imgDepth.at<unsigned short>(row, col) >
                                imgDepth.at<unsigned short>(ir + iir, ic + iic)) {
                                isMin = false;
                            }
                        }
                    }
                }

                isLocalMin.push_back(isMin);
            }
        }
    }




    bool isSuccess1 = false;
    bool isSuccess2 = false;

    double max = 100000000.0;

    if (!uvValues.empty() && !depths.empty()) {
        *glassEdges = Util::Unproject(uvValues, depths, intrinsic, distortion);

        for (int x = glassEdges->size() - 1; x > -1; x--) {
            if (glassEdges->at(x).x > max || glassEdges->at(x).y > max || glassEdges->at(x).z > max)
                glassEdges->erase(std::next(glassEdges->begin(), x));
            if (glassEdges->at(x).x == 0 || glassEdges->at(x).y == 0 || glassEdges->at(x).z == 0)
                glassEdges->erase(std::next(glassEdges->begin(), x));
        }




        if (glassEdges->size() > 100) {

            vector<Point3d> localMinEdges;

            for (int ge = 0; ge < glassEdges->size(); ge++) {
                if (isLocalMin[ge]) {
                    localMinEdges.push_back(glassEdges->at(ge));
                }
            }

            vector<int> rp;

            double inlierRate1 = 0.0;
            double inlierRate2 = 0.0;
            int iter = 0;

            while (!isInlier->empty()) {
                isInlier->erase(isInlier->begin());
            }
            for (int g = 0; g < glassEdges->size(); g++) {
                isInlier->push_back(false);
            }


            do {
                do {
                    for (int e = 0; e < numOfRp; e++) {
                        rp.push_back(rand() % localMinEdges.size());
                    }
                    if (!Util::HasSameValue(rp)) {
                        break;
                    } else {
                        while (!rp.empty()) {
                            rp.erase(rp.begin());
                        }
                    }
                } while (true);


                vector<bool> innerIsInlier;

                for (int g = 0; g < localMinEdges.size(); g++) {
                    innerIsInlier.push_back(false);
                }


                Point3d vec1 = localMinEdges[rp[1]] - localMinEdges[rp[0]];
                Point3d vec2 = localMinEdges[rp[2]] - localMinEdges[rp[0]];

                if (Util::IsParallel(vec1, vec2)) {
                    while (!rp.empty()) {
                        rp.erase(rp.begin());
                    }
                    iter++;
                    continue;
                }

                Point3d outerProduct = Point3d(vec1.y * vec2.z - vec1.z * vec2.y,
                                               vec1.z * vec2.x - vec1.x * vec2.z,
                                               vec1.x * vec2.y - vec1.y * vec2.x);

                double rh =
                        outerProduct.x * localMinEdges[rp[0]].x + outerProduct.y * localMinEdges[rp[0]].y +
                        outerProduct.z * localMinEdges[rp[0]].z;

                for (int f = 0; f < localMinEdges.size(); f++) {
                    double distance =
                            abs(outerProduct.x * localMinEdges[f].x + outerProduct.y * localMinEdges[f].y +
                                outerProduct.z * localMinEdges[f].z - rh)
                            / sqrt(outerProduct.x * outerProduct.x + outerProduct.y + outerProduct.y +
                                   outerProduct.z * outerProduct.z);
                    if (distance != NAN)
                        innerIsInlier[f] = distance < inlierThreshold;
                }

                int numOfInlierMin = 0;
                int numOfOutlierMin = 0;

                for (int f = 0; f < innerIsInlier.size(); f++) {
                    if (innerIsInlier[f]) {
                        numOfInlierMin++;
                    } else {
                        numOfOutlierMin++;
                    }
                }

                if (innerIsInlier.size() < 1) {
                    while (!rp.empty()) {
                        rp.erase(rp.begin());
                    }
                    iter++;
                    continue;
                }

                inlierRate1 = (double) numOfInlierMin / (double) innerIsInlier.size();

                if (inlierRate1 > dstInlierRate1) {
                    isSuccess1 = true;
                }
                if (isSuccess1) {
                    for (int f = 0; f < glassEdges->size(); f++) {
                        double distance =
                                abs(outerProduct.x * glassEdges->at(f).x + outerProduct.y * glassEdges->at(f).y +
                                    outerProduct.z * glassEdges->at(f).z - rh)
                                / sqrt(outerProduct.x * outerProduct.x + outerProduct.y + outerProduct.y +
                                       outerProduct.z * outerProduct.z);
                        if (distance != NAN) {
                            isInlier->at(f) = distance < inlierThreshold;
                        }
                    }
                    int numOfInlier = 0;
                    int numOfOutlier = 0;
                    for (int f = 0; f < isInlier->size(); f++) {
                        if (isInlier->at(f)) {
                            numOfInlier++;
                        } else {
                            numOfOutlier++;
                        }
                    }
                    inlierRate2 = (double) numOfInlier / (double) isInlier->size();
                    if (inlierRate2 > dstInlierRate2) {
                        isSuccess2 = true;

                        *outerProductRef = outerProduct;
                        *rhRef = rh;
                        return true;
                    }
                } else {
                    while (!rp.empty()) {
                        rp.erase(rp.begin());
                    }
                    iter++;
                    continue;
                }
                while (!rp.empty()) {
                    rp.erase(rp.begin());
                }
                iter++;
            } while (iter < maxIter);
        }
    }
    return false;
}