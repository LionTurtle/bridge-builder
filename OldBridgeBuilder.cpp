#include <stdio.h>
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class BridgeBuilder {
public:
    BridgeBuilder() {
        const int N_FEATS = 1000;
        detector = ORB(N_FEATS);

        matcher = DescriptorMatcher::create("BruteForce-Hamming");

        processMarkers();
    }

    // vector<KeyPoint> computeHomography(const Mat frame);
    void computeHomography(Mat frame);

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

// vector<KeyPoint> BridgeBuilder::computeHomography(const Mat frame) {
void BridgeBuilder::computeHomography(Mat frame) {

    vector<KeyPoint> fKPts;
    Mat fDesc;

    detector(frame, noArray(), fKPts, fDesc);

    vector< vector<DMatch> > matches1; // 2D array to store matches with marker 1
    vector< vector<DMatch> > matches2; // 2D array to store matches with marker 2
    matcher->knnMatch(mDesc1, fDesc, matches1, 2);
    matcher->knnMatch(mDesc2, fDesc, matches2, 2);

    // vector<KeyPoint> matchedInM1;
    // vector<KeyPoint> matchedToM1;

    // vector<KeyPoint> matchedInM2;
    // vector<KeyPoint> matchedToM2;

    vector<Point2f> matchedInM1;
    vector<int> indices;
    vector<Point2f> matchedToM1;

    // vector<Point2f> matchedInM2;
    // vector<Point2f> matchedToM2;

    const float MATCH_THRESH = 0.8f;
    for(unsigned i = 0; i < matches1.size(); i++) {
        if(matches1[i][0].distance < MATCH_THRESH * matches1[i][1].distance) {
            indices.push_back(i);
            matchedInM1.push_back(mKPts1[matches1[i][0].queryIdx].pt);
            matchedToM1.push_back(fKPts[matches1[i][0].trainIdx].pt);
            // matchedToM1.push_back(fKPts[matches1[i][0].trainIdx]);
        }
    }

    // drawKeypoints(frame, matchedToM1, frame, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
    // return matchedToM1;

    Mat homography;
    vector<unsigned char> inlierMask;
    const int RANSAC_THRESH = 5;
    if (matchedInM1.size() >= 4 && matchedInM1.size() == matchedToM1.size()) {
        homography = findHomography(Mat(matchedInM1), Mat(matchedToM1), RANSAC, RANSAC_THRESH, inlierMask);
    }

    // store inliers
    // vector<KeyPoint> inliersInM1;
    vector<KeyPoint> inliersToM1;
    // vector<DMatch> inlierMatches;
    // for(unsigned i = 0; i < matchedInM1.size(); i++) {
    for(unsigned i = 0; i < inlierMask.size(); i++) {
        // if(inlierMask.at<uchar>(i)) {
        if (inlierMask[i]) {
            // int new_i = static_cast<int>(inliersInM1.size());
            // inliersInM1.push_back(matchedInM1[i]);
            // inliersToM1.push_back(matchedToM1[i]);
            inliersToM1.push_back(fKPts[matches1[i][0].trainIdx]);
            // inlierMatches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    // drawKeypoints(frame, inliersToM1, frame, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
}  

int main(int argc, char *argv[]) {

    BridgeBuilder bb = BridgeBuilder();

    const int DEVICE_ID = 0;
    VideoCapture cap(DEVICE_ID);

    if(!cap.isOpened()) {
        cerr << "Capture Device ID cannot be opened." << endl;
        return -1;
    }

    namedWindow("BridgeBuilder", CV_WINDOW_AUTOSIZE);

    Mat frame;
    while (1) {

        cap >> frame;
        if (frame.empty()) break;

        bb.computeHomography(frame);
        // vector<KeyPoint> marker1 = bb.computeHomography(frame);
        // printf("nInliers = %d\n", (int)marker1.size());
        // drawKeypoints(frame, marker1, frame, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
        
        imshow("BridgeBuilder", frame);

        //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        if (waitKey(30) == 27) {
            break;
        }
    }

    return 0;
}