#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/asio.hpp>
#include "yolov8.hpp"
#include "BYTETracker.h"
#include "client.hpp"
#include "server.hpp"

// #define NO_PI //For testing with just a simple webcam.

bool end_program_flag = false;      //Not atomic since it does not mix well with cond_var.
std::condition_variable cond_var;   //Condition variable for UDP queue.
std::mutex m_udp;                   //Mutex for UDP queue
std::mutex m_data;                  //Mutex for counting data;
std::queue<cv::Mat> frame_queue;    //Queue of frames for UDP stream.

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

void udp_stream() {
    using namespace boost;

    const std::string ip = "239.255.0.1"; //Multicast address.
    constexpr unsigned port = 12345;
    const auto udp_ip = asio::ip::address::from_string(ip);
    constexpr int compression_factor = 80;
    std::vector<uint8_t> buff;
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY,compression_factor};

    asio::io_service io_service;
    asio::ip::udp::socket socket(io_service);
    asio::ip::udp::endpoint receiver_endpoint(udp_ip,port);
    system::error_code ec;

    socket.open(asio::ip::udp::v4());

    while(1) {
        std::unique_lock<std::mutex> lock{m_udp};
        cond_var.wait(lock,[](){return end_program_flag || !frame_queue.empty();});

        if(end_program_flag) {
            break;
        }

        cv::Mat frame = frame_queue.front();
        frame_queue.pop();
        lock.unlock();

        cv::imencode(".jpg",frame,buff,params);

        socket.send_to(asio::buffer(buff),receiver_endpoint,0,ec);

    }
}

void main_loop(Server& server) {

    //For some reason relative paths don't work
    const std::string classes_file_path = "/home/nvidia/Desktop/fishCenSV2/coco-classes.txt";
    const std::string engine_file_path = "/home/nvidia/Desktop/fishCenSV2/yolov8n.engine";

    //Address is the Jetson's. This should always be the same address as configured by the router.
    const std::string pipeline = "udpsrc address=192.168.0.103 port=5000 ! application/x-rtp, payload=96, encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! videoconvert ! video/x-raw, format=BGR, width=640, height=480 ! appsink";

    constexpr int fps = 50;

    //Right now it counts cellphones and anything else.
    //Of course this will be changed once the model is trained.
    int left_count = 0;
    int l_cell_count = 0;
    int right_count = 0;
    int r_cell_count = 0;

    cv::VideoCapture cap;
    cv::Mat frame;

    std::vector<Object> objects;                //Vector of bounding boxes
    std::vector<STrack> output_stracks;         //Vector of tracked objects
    std::unordered_map<int, int> previous_pos;  //Key = Track ID, Val = Previous Position

    Timer timer = Timer();         //Timer for measuring different processes execution time
    Timer timer_total = Timer();   //Timer for measuring total execution time

    YoloV8 detector(classes_file_path,engine_file_path);

    std::cout << "Initializing YoloV8 engine\n";

    detector.init();

    #ifdef NO_PI

    BYTETracker tracker(30,30);    

    if(!cap.open(1)) {
        std::cerr << "ERROR! Unable to open camera\n";
        return;
    }

    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);

    #else

    BYTETracker tracker(fps,30);    

    if(!cap.open(pipeline, cv::CAP_GSTREAMER)) {
        std::cerr << "ERROR! Unable to open camera\n";
        return;
    }

    #endif

    while(1) {

        timer_total.start();

        timer.start();

        //Sometimes `reference in DPB was never decoded` appears
        //This causes a massive delay and should be looked into further
        cap.read(frame);

        timer.end();

        std::cout << "Read Time: " << timer.get_time<Timer::microseconds>()/1000.0 <<"ms\n";

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
            for(int i = 0; i < output_stracks.size(); i++) {
                int current_position = static_cast<int>(.5 *(output_stracks[i].tlbr[0] + output_stracks[i].tlbr[2]));
                
                //If track_id isn't in the map then add it.
                if(previous_pos.count(output_stracks[i].track_id) == 0) {
                    previous_pos[output_stracks[i].track_id] = current_position;
                    continue;
                }

                else {
                    int previous_position = previous_pos[output_stracks[i].track_id];

                    if(current_position - previous_position > 0 && previous_position < 320 && current_position >= 320) {
                        if(detector.classes[objects[i].label] == "cell phone") {
                            r_cell_count++;
                        }

                        else {
                            right_count++;
                        }

                    }

                    else if(current_position - previous_position < 0 && previous_position >= 320 && current_position < 320) {
                        if(detector.classes[objects[i].label] == "cell phone") {
                            l_cell_count++;
                        }

                        else {
                            left_count++;
                        }
                    }

                    previous_pos[output_stracks[i].track_id] = current_position;
                }

                //Code below directly grabbed from one of the ByteTrack TensorRT files
                std::vector<float> tlwh = output_stracks[i].tlwh;
                float prob_score = objects[i].prob * 100;;

                cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
                cv::putText(frame, cv::format("%s#%d", (detector.classes[objects[i].label]).c_str(),output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1]-24),
                0, 0.6, cv::Scalar(0,0,255),1,cv::LINE_AA);
                cv::putText(frame, cv::format("%.2f%s",prob_score,"%"), cv::Point(tlwh[0], tlwh[1]-5),
                0, 0.6, cv::Scalar(0,0,255),1,cv::LINE_AA);
                cv::rectangle(frame, cv::Rect(tlwh[0],tlwh[1],tlwh[2],tlwh[3]),s,2);
                
            }

            {
                std::lock_guard<std::mutex> l(m_data);
                server.data = {l_cell_count, r_cell_count, left_count, right_count};
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
        cv::putText(frame,cv::format("Right Cell Count = %d", r_cell_count),cv::Point(30,90),0,1,cv::Scalar(255,255,255));
        cv::putText(frame,cv::format("Left Cell Count = %d", l_cell_count),cv::Point(30,120),0,1,cv::Scalar(255,255,255));

        cv::line(frame,cv::Point(319,0),cv::Point(319,639),cv::Scalar(0,255,0),3);

        timer.end();

        std::cout << "Draw Time: " << timer.get_time<Timer::microseconds>()/1000.0 <<"ms\n";

        cv::imshow("Live", frame);

        {
            std::lock_guard<std::mutex> l{m_udp};
            frame_queue.push(frame);
        }

        //Notify UDP thread
        cond_var.notify_one();

        timer_total.end();

        std::cout << "Total Time: " << timer_total.get_time<Timer::milliseconds>() <<"ms\n";

        if(cv::waitKey(1) >= 0) {
            std::lock_guard<std::mutex> lock(m_udp);
            end_program_flag = true;
            cond_var.notify_one();
            server.stop();
            break;
        }

    }
}

int main() {
    boost::asio::io_context io_context;
    Server server(io_context, 1234, m_data);

    std::thread server_th([&]() {server.run();});
    std::thread udp_stream_th(udp_stream);
    std::thread main_loop_th(main_loop, std::ref(server));

    server_th.join();
    udp_stream_th.join();
    main_loop_th.join();

    return 0;
}


