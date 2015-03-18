#include <stdio.h>
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

void processMarkers(ORB orbDetector, vector<KeyPoint> kpts1, vector<KeyPoint> kpts2, Mat desc1, Mat desc2) {
    Mat marker1 = imread("../markers/marker1.png", 1);
    Mat marker2 = imread("../markers/marker2.png", 1);

    const Size MARKER_SIZE = Size(500, 500);
    resize(marker1, marker1, MARKER_SIZE);
    resize(marker2, marker2, MARKER_SIZE);

    orbDetector(marker1, noArray(), kpts1, desc1);
    orbDetector(marker2, noArray(), kpts2, desc2);

    Mat imgWithKPts1;
    Mat imgWithKPts2;
    drawKeypoints(marker1, kpts1, imgWithKPts1, Scalar(255, 0, 0), DrawMatchesFlags::DEFAULT);
    drawKeypoints(marker2, kpts2, imgWithKPts2, Scalar(255, 0, 0), DrawMatchesFlags::DEFAULT);

    imshow("marker1", imgWithKPts1);
    imshow("marker2", imgWithKPts2);

    printf("n m1 kpts: %d\n", (int)kpts1.size());
    printf("n m2 kpts: %d\n", (int)kpts2.size());

    waitKey(0);
}

int main(int argc, char *argv[]) {

    const int N_FEATS = 150;
    ORB orb = ORB(N_FEATS);

    vector<KeyPoint> mKPts1;
    vector<KeyPoint> mKPts2;
    Mat mDesc1;
    Mat mDesc2;

    processMarkers(orb, mKPts1, mKPts2, mDesc1, mDesc2);

}