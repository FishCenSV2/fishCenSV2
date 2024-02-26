#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "yolov8.hpp"
#include "BYTETracker.h"

using milliseconds = std::chrono::milliseconds;

class Timer {
    private:
        std::chrono::high_resolution_clock::time_point _start_time;
        std::chrono::high_resolution_clock::time_point _end_time;
    public:
        inline void start() noexcept {
            _start_time = std::chrono::high_resolution_clock::now();
        };
        inline void end() noexcept {
            _end_time = std::chrono::high_resolution_clock::now();
        };
        template<typename T>
        inline long long get_time() {
            return std::chrono::duration_cast<T>(_end_time-_start_time).count();
        };
};

int main() {
    std::string classes_file_path = "/home/nvidia/Desktop/fishCenSV2/coco-classes.txt";
    std::string engine_file_path = "/home/nvidia/Desktop/fishCenSV2/yolov8n.engine";

    YoloV8 detector(classes_file_path,engine_file_path);
    detector.init();

    cv::Mat frame;
    cv::VideoCapture cap;

    cap.open(1);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if(!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    std::vector<Object> objects;

    Timer timer = Timer();         //Timer for measuring different processes execution time
    Timer timer_total = Timer();   //Timer for measuring total execution time

    BYTETracker tracker(30,30);

    while(1) {

        timer_total.start();

        timer.start();

        cap.read(frame);

        timer.end();

        std::cout << "Read Time: " << timer.get_time<milliseconds>() <<"ms\n";

        if(frame.empty()) {
            std::cerr << "ERROR! Blank frame grabbed\n";
            break;
        }

        timer.start();

        cv::Mat input = detector.preprocess(frame);

        timer.end();

        std::cout << "Preprocess Time: " << timer.get_time<milliseconds>() <<"ms\n";

        timer.start();

        detector.predict(input);

        timer.end();

        std::cout << "Inference Time: " << timer.get_time<milliseconds>() <<"ms\n";

        timer.start();

        objects = detector.postprocess(2);

        timer.end();

        std::cout << "Postprocess Time: " << timer.get_time<milliseconds>() <<"ms\n";

        cv::copyMakeBorder(frame,frame, 80,80,0,0, CV_HAL_BORDER_CONSTANT, cv::Scalar(0,0,0));

        timer.start();

        std::vector<STrack> output_stracks = tracker.update(objects);

        timer.end();

        std::cout << "Tracker Time: " << timer.get_time<milliseconds>() <<"ms\n";

        timer.start();

        //Code directly grabbed from one of the ByteTrack TensorRT files.
        //Some of this will be changed as I believe the way it is written
        //it won't plot some boxes if they aren't "people"-sized. 
        for(int i = 0; i < output_stracks.size(); i++) {
            std::vector<float> tlwh = output_stracks[i].tlwh;
            bool vertical = tlwh[2] / tlwh[3] > 1.6;
            if(tlwh[2] * tlwh[3] > 20 && !vertical) {
                cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
                cv::putText(frame, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1]-5),
                0, 0.6, cv::Scalar(0,0,255),2,cv::LINE_AA);
                cv::rectangle(frame, cv::Rect(tlwh[0],tlwh[1],tlwh[2],tlwh[3]),s,2);
            }
        }

        timer.end();

        std::cout << "Draw Time: " << timer.get_time<milliseconds>() <<"ms\n";

        cv::imshow("Live", frame);

        timer_total.end();

        std::cout << "Total Time: " << timer_total.get_time<milliseconds>() <<"ms\n";

        if(cv::waitKey(1) >= 0) {
            break;
        }

    }


    return 0;
}


