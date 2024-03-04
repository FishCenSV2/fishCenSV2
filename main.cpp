#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "yolov8.hpp"
#include "BYTETracker.h"

class Timer {
    private:
        std::chrono::high_resolution_clock::time_point _start_time;
        std::chrono::high_resolution_clock::time_point _end_time;
    public:
        using seconds = std::chrono::seconds;
        using milliseconds = std::chrono::milliseconds;
        using microseconds = std::chrono::microseconds;
        using nanoseconds = std::chrono::nanoseconds;

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

    std::vector<STrack> output_stracks;
    std::unordered_map<int, int> previous_pos; //Key = Track ID, Val = Previous Position

    int left_count = 0;
    int right_count = 0;

    while(1) {

        timer_total.start();

        timer.start();

        cap.read(frame);

        timer.end();

        std::cout << "Read Time: " << timer.get_time<Timer::milliseconds>() <<"ms\n";

        if(frame.empty()) {
            std::cerr << "ERROR! Blank frame grabbed\n";
            break;
        }

        timer.start();

        cv::Mat input = detector.preprocess(frame);

        timer.end();

        std::cout << "Preprocess Time: " << timer.get_time<Timer::milliseconds>() <<"ms\n";

        timer.start();

        detector.predict(input);

        timer.end();

        std::cout << "Inference Time: " << timer.get_time<Timer::milliseconds>() <<"ms\n";

        timer.start();

        objects = detector.postprocess(2);

        timer.end();

        std::cout << "Postprocess Time: " << timer.get_time<Timer::microseconds>()/1000.0 <<"ms\n";

        cv::copyMakeBorder(frame,frame, 80,80,0,0, CV_HAL_BORDER_CONSTANT, cv::Scalar(0,0,0));

        timer.start();

        output_stracks = tracker.update(objects);

        timer.end();

        std::cout << "Tracker Time: " << timer.get_time<Timer::microseconds>()/1000.0 <<"ms\n";

        timer.start();

        if(output_stracks.size() > 0) {
            
            //Loop over all output tracks and check if they crossed the boundary.
            //NOTE: Still need to add code to remove older track ids otherwise we
            //      could get a memory leak. Current idea is to remove them after
            //      a certain time duration.
            for(int i = 0; i < output_stracks.size(); i++) {
                int current_position = static_cast<int>(.5 *(output_stracks[i].tlbr[0] + output_stracks[i].tlbr[2]));
                
                //If track_id isn't in the map then add it.
                if(previous_pos.count(output_stracks[i].track_id) == 0) {
                    previous_pos[output_stracks[i].track_id] = current_position;
                    continue;
                }

                else {
                    int previous_position = previous_pos[output_stracks[i].track_id];

                    if(current_position - previous_position > 0 && previous_position < 340 && current_position >= 340) {
                        right_count++;
                    }

                    else if(current_position - previous_position < 0 && previous_position >= 340 && current_position < 340) {
                        left_count++;
                    }

                    previous_pos[output_stracks[i].track_id] = current_position;
                }

                //Code below directly grabbed from one of the ByteTrack TensorRT files
                std::vector<float> tlwh = output_stracks[i].tlwh;

                cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
                cv::putText(frame, cv::format("%s #%d", detector.classes[objects[i].label].c_str(),output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1]-5),
                0, 0.6, cv::Scalar(0,0,255),1,cv::LINE_AA);
                cv::rectangle(frame, cv::Rect(tlwh[0],tlwh[1],tlwh[2],tlwh[3]),s,2);
                
            }

        }

        cv::putText(frame,cv::format("Right Count = %d", right_count),cv::Point(30,30),0,1,cv::Scalar(255,255,255));
        cv::putText(frame,cv::format("Left Count = %d", left_count),cv::Point(30,60),0,1,cv::Scalar(255,255,255));

        cv::line(frame,cv::Point(319,0),cv::Point(319,639),cv::Scalar(0,255,0),3);

        timer.end();

        std::cout << "Draw Time: " << timer.get_time<Timer::microseconds>()/1000.0 <<"ms\n";

        cv::imshow("Live", frame);

        timer_total.end();

        std::cout << "Total Time: " << timer_total.get_time<Timer::milliseconds>() <<"ms\n";

        if(cv::waitKey(1) >= 0) {
            break;
        }

    }


    return 0;
}


