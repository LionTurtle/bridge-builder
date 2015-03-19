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

void BridgeBuilder::computeHomography(Mat frame) {

    // detect keypoints and descriptors in the current frame
    vector<KeyPoint> fKPts;
    Mat fDesc;
    detector(frame, noArray(), fKPts, fDesc);

    // match descriptors from the current frame to the descriptors of the two markers
    vector< vector<DMatch> > matches1; // 2D array to store matches with marker 1
    vector< vector<DMatch> > matches2; // 2D array to store matches with marker 2
    matcher->knnMatch(mDesc1, fDesc, matches1, 2);
    matcher->knnMatch(mDesc2, fDesc, matches2, 2);

    // use Lowe's nearest/2nd nearest neighbor to find best candidate match for keypoints
    vector<KeyPoint> queryKPts1;
    vector<KeyPoint> trainKPts1;
    vector<KeyPoint> queryKPts2;
    vector<KeyPoint> trainKPts2;
    const float MATCH_THRESH = 0.8f;
    for(unsigned i = 0; i < matches1.size(); i++) {
        if(matches1[i][0].distance < MATCH_THRESH * matches1[i][1].distance) {
            queryKPts1.push_back(mKPts1[matches1[i][0].queryIdx]);
            trainKPts1.push_back(fKPts[matches1[i][0].trainIdx]);
        }
        
    }
    for (unsigned  i = 0; i < matches2.size(); i++) {
        if(matches2[i][0].distance < MATCH_THRESH * matches2[i][1].distance) {
            queryKPts2.push_back(mKPts2[matches2[i][0].queryIdx]);
            trainKPts2.push_back(fKPts[matches2[i][0].trainIdx]);
        }
    }

    // now that we have only the best candidates from matches,
    // get pts from keypoints to use in findHomography
    vector<Point2f> queryPts1;
    vector<Point2f> trainPts1;
    vector<Point2f> queryPts2;
    vector<Point2f> trainPts2;
    for (unsigned i = 0; i < queryKPts1.size(); i++) {
        queryPts1.push_back(queryKPts1[i].pt);
        trainPts1.push_back(trainKPts1[i].pt);
        
    }
    for (unsigned i = 0; i < queryKPts2.size(); i++) {
        queryPts2.push_back(queryKPts2[i].pt);
        trainPts2.push_back(trainKPts2[i].pt);
    }

    // refine keypoint matches using RANSAC
    Mat homography1;
    Mat homography2;
    vector<unsigned char> inlierMask1;
    vector<unsigned char> inlierMask2;
    const int RANSAC_THRESH = 5;
    if (queryPts1.size() >= 4 && trainPts1.size() == queryPts1.size()) {
        homography1 = findHomography(Mat(queryPts1), Mat(trainPts1), RANSAC, RANSAC_THRESH, inlierMask1);
    }
    if (queryPts2.size() >= 4 && trainPts2.size() == queryPts2.size()) {
        homography2 = findHomography(Mat(queryPts2), Mat(trainPts2), RANSAC, RANSAC_THRESH, inlierMask2);
    }

    // get inlier keypoints of this model
    vector<KeyPoint> inliers1;
    vector<KeyPoint> inliers2;
    for(unsigned i = 0; i < inlierMask1.size(); i++) {
        if (inlierMask1[i]) {
            inliers1.push_back(trainKPts1[i]);
        }
    }
    for(unsigned i = 0; i < inlierMask2.size(); i++) {
        if (inlierMask2[i]) {
            inliers2.push_back(trainKPts2[i]);
        }
    }

    drawKeypoints(frame, inliers1, frame, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
    drawKeypoints(frame, inliers2, frame, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
    // printf("nInliers = %d\n", (int)inliers1.size());
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
        imshow("BridgeBuilder", frame);

        //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        if (waitKey(30) == 27) break;
    }

    return 0;
}