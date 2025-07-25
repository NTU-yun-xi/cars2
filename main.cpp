#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

const int min_w = 90;
const int min_h = 90;
const int line_high = 490;
const int offset = 8;
int carno = 0;
vector<Point> cars;

Point center(int x, int y, int w, int h) {
    int cx = x + w / 2;
    int cy = y + h / 2;
    return Point(cx, cy);
}

int main() {
    VideoCapture cap("D:/pyproject/cheliu/carvideo.mp4");
    if (!cap.isOpened()) {
        cerr << "无法打开视频文件" << endl;
        return -1;
    }

    Ptr<BackgroundSubtractor> bgsubmog = createBackgroundSubtractorMOG2();
    Mat kernel = getStructuringElement(MORPH_RECT, Size(14, 14));

    Mat frame, gray, blur, mask, erode_mat, dilate_mat, close_mat;  // 避免使用关键字作为变量名
    vector<vector<Point>> cnts;
    vector<Vec4i> hierarchy;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blur, Size(3, 3), 5);
        bgsubmog->apply(blur, mask);

        // 修正：形态学操作的参数和变量名
        erode(mask, erode_mat, kernel);
        dilate(erode_mat, dilate_mat, kernel, Point(-1, -1), 2);  // 这里修正了参数传递方式
        morphologyEx(dilate_mat, close_mat, MORPH_CLOSE, kernel);
        morphologyEx(close_mat, close_mat, MORPH_CLOSE, kernel);

        findContours(close_mat, cnts, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        line(frame, Point(10, line_high), Point(1000, line_high), Scalar(0, 255, 0), 2);

        for (size_t i = 0; i < cnts.size(); i++) {
            Rect rect = boundingRect(cnts[i]);
            if (rect.width >= min_w && rect.height >= min_h) {
                rectangle(frame, rect, Scalar(255, 0, 255), 2);
                Point cpoint = center(rect.x, rect.y, rect.width, rect.height);
                cars.push_back(cpoint);

                for (auto it = cars.begin(); it != cars.end();) {
                    if (it->y > line_high - offset && it->y < line_high + offset) {
                        carno++;
                        it = cars.erase(it);
                        cout << carno << endl;
                    }
                    else {
                        ++it;
                    }
                }
            }
        }

        putText(frame, "CARS COUNT: " + to_string(carno), Point(500, 60),
            FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 5);
        imshow("video", frame);

        char key = waitKey(1);
        if (key == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}