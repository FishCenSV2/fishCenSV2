#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/asio.hpp>
#include "yolov8.hpp"
#include "BYTETracker.h"
#include "client.hpp"

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

    boost::asio::io_context io_context;
    constexpr int height = 480;
    constexpr int width = 640;
    constexpr int color_channels = 3;
    constexpr size_t buff_size = height * width * color_channels;

    std::vector<std::uint8_t> buff(buff_size);
    unsigned request = 5; //Will later change to be an enum possibly.

    //Might be possible that client connects before server can start?
    //Need to investigate further.
    Client client(io_context,"192.168.1.85",1234);
    client.connect();

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

        client.write(request);
        client.read(buff);

        cv::Mat frame = cv::Mat(height,width,CV_8UC3,buff.data());

        timer.end();

        std::cout << "Read Time: " << timer.get_time<Timer::milliseconds>() <<"ms\n";

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

        //Loops over all removed tracks and removes corresponding track IDs 
        //from the `previous_pos` map
        if(tracker.removed_stracks.size() > 0) {

            int temp_track_id = 0;

            for (auto &i: tracker.removed_stracks) {

                //For some reason the removed tracks vector can have duplicates
                if(i.track_id != temp_track_id) {
                    previous_pos.erase(i.track_id);
                }

                temp_track_id = i.track_id;
            }

            tracker.removed_stracks.clear();
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


