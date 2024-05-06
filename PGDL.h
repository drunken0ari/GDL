//
// Created by hanta on 24. 4. 30.
//

#ifndef GLASSPLANEDETECTIONANDLOCALIZATION_GDL_H
#define GLASSPLANEDETECTIONANDLOCALIZATION_GDL_H

#include "Util.h"

class GDL {
public:
    static void SeparateBinaryImageSegments(Mat imgSeg, Mat& separatedSeg);
    static void MakeEdgeImage(Mat imgSeg, Mat separatedSeg, Mat& imgEdge);
    static void ExtractCandidatesDepthAware(Mat imgDepth, Mat imgEdge, Mat& imgEdgeCddt, double distThreshold, int windowSize);
    static bool EstimatePlaneBySegment(const int windowSize, Mat imgEdgeCddt, int segmentNum, Mat imgDepth
            , vector<Point3d>* glassEdges, vector<bool>* isInlier
            , const Mat intrinsic, const Mat distortion, int numOfRp
            , double inlierThreshold, double dstInlierRate1, double dstInlierRate2, int maxIter
            , Point3d* outerProductRef, double* rhRef);
};


#endif //GLASSPLANEDETECTIONANDLOCALIZATION_GDL_H
