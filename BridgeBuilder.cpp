#include <stdio.h>
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class BridgeBuilder {
public:
    BridgeBuilder() {
        const int N_FEATS = 150;
        detector = ORB(N_FEATS);

        matcher = DescriptorMatcher::create("BruteForce-Hamming");

        processMarkers();
    }

    void computeHomography(const Mat frame);

protected:
    ORB detector;
    Ptr<DescriptorMatcher> matcher;
    vector<KeyPoint> mKPts1;
    vector<KeyPoint> mKPts2;
    Mat mDesc1;
    Mat mDesc2;

private:
    void processMarkers();
};

void BridgeBuilder::processMarkers() {
    Mat marker1 = imread("../markers/marker1.png", 1);
    Mat marker2 = imread("../markers/marker2.png", 1);

    const Size MARKER_SIZE = Size(500, 500);
    resize(marker1, marker1, MARKER_SIZE);
    resize(marker2, marker2, MARKER_SIZE);

    detector(marker1, noArray(), mKPts1, mDesc1);
    detector(marker2, noArray(), mKPts2, mDesc2);

    // Mat imgWithKPts1;
    // Mat imgWithKPts2;
    // drawKeypoints(marker1, mKPts1, imgWithKPts1, Scalar(255, 0, 0), DrawMatchesFlags::DEFAULT);
    // drawKeypoints(marker2, mKPts2, imgWithKPts2, Scalar(255, 0, 0), DrawMatchesFlags::DEFAULT);

    // imshow("marker1", imgWithKPts1);
    // imshow("marker2", imgWithKPts2);

    // printf("n m1 kpts: %d\n", (int)mKPts1.size());
    // printf("n m2 kpts: %d\n", (int)mKPts2.size());

    // waitKey(0);
}

void BridgeBuilder::computeHomography(const Mat frame) {

    vector<KeyPoint> fKPts;
    Mat fDesc;

    detector(frame, noArray(), fKPts, fDesc);

    vector< vector<DMatch> > matches1; // 2D array to store matches with marker 1
    vector< vector<DMatch> > matches2; // 2D array to store matches with marker 2
    matcher->knnMatch(mDesc1, fDesc, matches1, 2);
    matcher->knnMatch(mDesc2, fDesc, matches2, 2);

    vector<KeyPoint> matchedToM1;
    vector<KeyPoint> matchedInM1;

    vector<KeyPoint> matchedToM2;
    vector<KeyPoint> matchedInM2;

    const float MATCH_THRESH = 0.7f;
    for(int i = 0; i < matches1.size(); i++) {
        float ratio = (float)matches1[i][0].distance / (float)matches1[i][1].distance;
        if(ratio < MATCH_THRESH) {
            matchedInM1.push_back(mKPts1[matches1[i][0].queryIdx]);
            matchedToM1.push_back(fKPts[matches1[i][0].trainIdx]);
        }
    }

}

int main(int argc, char *argv[]) {

    BridgeBuilder bb = BridgeBuilder();

}