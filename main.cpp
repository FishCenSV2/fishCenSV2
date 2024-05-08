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
#include <boost/circular_buffer.hpp>
#include "yolov8.hpp"
#include "BYTETracker.h"
#include "client.hpp"
#include "server.hpp"
#include "util.hpp"

// #define NO_PI //For testing with just a simple webcam.
// #define DEBUG_MODE
#define MAX_READ_QUEUE_SIZE 30

bool end_program_flag = false;                  //Not atomic since it does not mix well with cond_var.
std::condition_variable cond_udp_var;           //Condition variable for UDP queue.
std::condition_variable cond_read_var;          //Condition variable for frame reading queue
std::mutex m_udp;                               //Mutex for UDP queue.
std::mutex m_data;                              //Mutex for counting data.
std::mutex m_read;                              //Mutex for frame reading queue.
std::queue<cv::Mat> frame_queue;                //Queue of frames for UDP stream.

//Queue of read frames from camera.
std::queue<cv::Mat,boost::circular_buffer<cv::Mat>> read_frame_queue(boost::circular_buffer<cv::Mat>(MAX_READ_QUEUE_SIZE));

#ifdef DEBUG_MODE
std::vector<float> read_times;
std::vector<float> preprocess_times;
std::vector<float> inference_times;
std::vector<float> postprocess_times;
std::vector<float> tracker_times;
#endif

void read_frame() {
    //Address is the Jetson's. This should always be the same address as configured by the router.
    const std::string pipeline = "udpsrc address=192.168.0.103 port=5000 ! application/x-rtp, payload=96, encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! videoconvert ! video/x-raw, format=BGR, width=640, height=480 ! appsink";

    cv::VideoCapture cap;
    cv::Mat frame;

    #ifdef NO_PI

    if(!cap.open(0)) {
        std::cerr << "ERROR! Unable to open camera\n";
        return;
    }

    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);

    #else

    if(!cap.open(pipeline, cv::CAP_GSTREAMER)) {
        std::cerr << "ERROR! Unable to open camera\n";
        return;
    }

    #endif

    std::cout << "Connected to camera\n";

    while(1) {

        cap.read(frame);

        {
            std::lock_guard<std::mutex> l{m_read};
            read_frame_queue.push(frame);
        }

        //Notify main_loop thread 
        cond_read_var.notify_one();

        if(end_program_flag) {
            break;
        }

    }
}

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
        cond_udp_var.wait(lock,[](){return end_program_flag || !frame_queue.empty();});

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
    const std::string classes_file_path = "/home/nvidia/Desktop/fishCenSV2/coco-classes-fish.txt";
    const std::string engine_file_path = "/home/nvidia/Desktop/fishCenSV2/best_t21.engine";

    constexpr int fps = 20;

    //Counts of fish swimming left
    int left_count = 0;
    int l_chinook_count = 0;
    int l_chum_count = 0;
    int l_coho_count = 0;
    int l_sockeye_count = 0;
    int l_steelhead_count = 0;

    //Counts of fish swimming right
    int right_count = 0;
    int r_chinook_count = 0;
    int r_chum_count = 0;
    int r_coho_count = 0;
    int r_sockeye_count = 0;
    int r_steelhead_count = 0;

    std::vector<Object> objects;                //Vector of bounding boxes
    std::vector<STrack> output_stracks;         //Vector of tracked objects
    std::unordered_map<int, int> previous_pos;  //Key = Track ID, Val = Previous Position

    util::StopWatch stopwatch = util::StopWatch();         //Timer for measuring different processes execution time
    util::StopWatch stopwatch_total = util::StopWatch();   //Timer for measuring total execution time

    YoloV8 detector(classes_file_path,engine_file_path);

    std::cout << "Initializing YoloV8 engine\n";

    detector.init();

    #ifdef NO_PI

    BYTETracker tracker(30,30);    

    #else

    BYTETracker tracker(fps,30);    

    #endif

    while(1) {
        stopwatch_total.start();

        stopwatch.start();

        std::unique_lock<std::mutex> lock{m_read};
        cond_read_var.wait(lock,[](){return end_program_flag || !read_frame_queue.empty();});

        cv::Mat frame = read_frame_queue.front();
        read_frame_queue.pop();
        lock.unlock();

        stopwatch.end();

        std::cout << "Read Time: " << stopwatch.get_time<util::StopWatch::microseconds>()/1000.0 <<"ms\n";

        #ifdef DEBUG_MODE
            read_times.push_back(stopwatch.get_time<util::StopWatch::microseconds>()/1000.0);
        #endif

        if(frame.empty()) {
            std::cerr << "ERROR! Blank frame grabbed\n";
            break;
        }

        stopwatch.start();

        cv::Mat input = detector.preprocess(frame);

        stopwatch.end();

        std::cout << "Preprocess Time: " << stopwatch.get_time<util::StopWatch::milliseconds>() <<"ms\n";

        #ifdef DEBUG_MODE
            preprocess_times.push_back(stopwatch.get_time<util::StopWatch::milliseconds>());
        #endif

        stopwatch.start();

        detector.predict(input);

        stopwatch.end();

        std::cout << "Inference Time: " << stopwatch.get_time<util::StopWatch::milliseconds>() <<"ms\n";
        
        #ifdef DEBUG_MODE
            inference_times.push_back(stopwatch.get_time<util::StopWatch::milliseconds>());
        #endif

        stopwatch.start();

        //Scale up bounding boxes by 1. Input was 640x640 and output is also 640x640.
        //Output is what is shown on the OpenCV
        objects = detector.postprocess(1);

        std::cout << "# objects before tracker: " << objects.size() << std::endl;

        stopwatch.end();

        std::cout << "Postprocess Time: " << stopwatch.get_time<util::StopWatch::microseconds>()/1000.0 <<"ms\n";

        #ifdef DEBUG_MODE
            postprocess_times.push_back(stopwatch.get_time<util::StopWatch::microseconds>()/1000.0);
        #endif

        cv::copyMakeBorder(frame,frame, 80,80,0,0, CV_HAL_BORDER_CONSTANT, cv::Scalar(0,0,0));

        stopwatch.start();

        output_stracks = tracker.update(objects);

        stopwatch.end();

        std::cout << "Tracker Time: " << stopwatch.get_time<util::StopWatch::microseconds>()/1000.0 <<"ms\n";

        #ifdef DEBUG_MODE
            tracker_times.push_back(stopwatch.get_time<util::StopWatch::microseconds>()/1000.0);
        #endif

        stopwatch.start();

        #ifdef DEBUG_MODE
        std::cout << "# output_stracks after postprocess: " << output_stracks.size() << std::endl;
        #endif

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
                        if(detector.classes[objects[i].label] == "Chinook") {
                            r_chinook_count++;
                        }

                        else if(detector.classes[objects[i].label] == "Chum") {
                            r_chum_count++;
                        }

                        else if(detector.classes[objects[i].label] == "Coho") {
                            r_coho_count++;
                        } 

                        else if(detector.classes[objects[i].label] == "Sockeye") {
                            r_sockeye_count++;
                        }

                        else if(detector.classes[objects[i].label] == "Steelhead") {
                            r_steelhead_count++;
                        }

                        right_count++;
                    }

                    else if(current_position - previous_position < 0 && previous_position >= 320 && current_position < 320) {
                        if(detector.classes[objects[i].label] == "Chinook") {
                            l_chinook_count++;
                        }

                        else if(detector.classes[objects[i].label] == "Chum") {
                            l_chum_count++;
                        }

                        else if(detector.classes[objects[i].label] == "Coho") {
                            l_coho_count++;
                        } 

                        else if(detector.classes[objects[i].label] == "Sockeye") {
                            l_sockeye_count++;
                        }

                        else if(detector.classes[objects[i].label] == "Steelhead") {
                            l_steelhead_count++;
                        }

                        left_count++;
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
                server.data = {l_chinook_count, r_chinook_count, l_chum_count, r_chum_count, l_coho_count, r_coho_count, 
                                l_sockeye_count, r_sockeye_count, l_steelhead_count, r_steelhead_count, left_count, right_count};
            }

        }

        #ifdef DEBUG_MODE
        std::cout << "# output_stracks after boundary check: " << output_stracks.size() << std::endl;
        #endif

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

        #ifdef DEBUG_MODE
        //std::cout << "# output_stracks after track removal: " << output_stracks.size() << std::endl;
        #endif

        cv::putText(frame,cv::format("Right Count = %d", right_count),cv::Point(30,30),0,1,cv::Scalar(255,255,255));
        cv::putText(frame,cv::format("Left Count = %d", left_count),cv::Point(30,60),0,1,cv::Scalar(255,255,255));

        cv::line(frame,cv::Point(319,0),cv::Point(319,639),cv::Scalar(0,255,0),3);

        stopwatch.end();

        std::cout << "Draw Time: " << stopwatch.get_time<util::StopWatch::microseconds>()/1000.0 <<"ms\n";

        cv::resize(frame, frame, Size(1100, 1100), 0, 0, INTER_AREA);

        cv::imshow("Live", frame);

        {
            std::lock_guard<std::mutex> l{m_udp};
            frame_queue.push(frame);
        }

        //Notify UDP thread
        cond_udp_var.notify_one();

        stopwatch_total.end();

        std::cout << "Total Time: " << stopwatch_total.get_time<util::StopWatch::milliseconds>() <<"ms\n";

        if(cv::waitKey(1) >= 0) {
            std::lock_guard<std::mutex> lock(m_udp);
            end_program_flag = true;
            cond_udp_var.notify_one();
            server.stop();
            break;
        }

    }

}

int main() {

    boost::asio::io_context io_context;
    Server server(io_context, 1234, m_data);

    std::thread read_frame_th(read_frame);
    std::thread server_th([&]() {server.run();});
    std::thread udp_stream_th(udp_stream);
    std::thread main_loop_th(main_loop, std::ref(server));

    read_frame_th.join();
    server_th.join();
    udp_stream_th.join();
    main_loop_th.join();

    std::cout << "\nLeft Counts\n";
    std::cout << "Chinook: " << server.data[0] << "\n";
    std::cout << "Chum: " << server.data[2] << "\n"; 
    std::cout << "Coho: " << server.data[4] << "\n"; 
    std::cout << "Sockeye: " << server.data[6] << "\n"; 
    std::cout << "Steelhead: " << server.data[8] << "\n"; 
    std::cout << "Total: " <<  server.data[10] << "\n\n";

    std::cout << "Right Counts\n";
    std::cout << "Chinook: " << server.data[1] << "\n";
    std::cout << "Chum: " << server.data[3] << "\n"; 
    std::cout << "Coho: " << server.data[5] << "\n"; 
    std::cout << "Sockeye: " << server.data[7] << "\n"; 
    std::cout << "Steelhead: " << server.data[9] << "\n"; 
    std::cout << "Total: " <<  server.data[11] << "\n\n";
    
    #ifdef DEBUG_MODE

    auto remove_large_times = [&](float i) -> bool {
        return i > 300;
    };

    auto read_times_it = std::remove_if(read_times.begin(), read_times.end(), remove_large_times);

    read_times.erase(read_times_it, read_times.end());

    std::cout << "Average Read Time: " << util::accumulate(read_times.begin(),read_times.end(),0.0) / static_cast<float>(read_times.size())<< "ms\n";
    std::cout << "Max Read Time: " << *std::max_element(read_times.begin(),read_times.end()) << "ms\n";
    std::cout << "Min Read Time: " << *std::min_element(read_times.begin(),read_times.end()) << "ms\n";

    preprocess_times.erase(preprocess_times.begin()+20);

    std::cout << "Average Preprocess Time: " << util::accumulate(preprocess_times.begin(),preprocess_times.end(),0.0) / static_cast<float>(preprocess_times.size()) << "ms\n";
    std::cout << "Max Preprocess Time: " << *std::max_element(preprocess_times.begin(),preprocess_times.end()) << "ms\n";
    std::cout << "Min Preprocess Time: " << *std::min_element(preprocess_times.begin(),preprocess_times.end()) << "ms\n";

    auto inference_times_it = std::remove_if(inference_times.begin(), inference_times.end(), remove_large_times);

    inference_times.erase(inference_times_it, inference_times.end());

    std::cout << "Average Inference Time: " << util::accumulate(inference_times.begin(),inference_times.end(),0.0) / static_cast<float>(inference_times.size()) << "ms\n";
    std::cout << "Max Inference Time: " << *std::max_element(inference_times.begin(),inference_times.end()) << "ms\n";
    std::cout << "Min Inference Time: " << *std::min_element(inference_times.begin(),inference_times.end()) << "ms\n";

    postprocess_times.erase(postprocess_times.begin()+20);

    std::cout << "Average Postprocess Time: " << util::accumulate(postprocess_times.begin(),postprocess_times.end(),0.0) / static_cast<float>(postprocess_times.size()) << "ms\n";
    std::cout << "Max Postprocess Time: " << *std::max_element(postprocess_times.begin(),postprocess_times.end()) << "ms\n";
    std::cout << "Min Postprocess Time: " << *std::min_element(postprocess_times.begin(),postprocess_times.end()) << "ms\n";

    tracker_times.erase(tracker_times.begin()+20);

    std::cout << "Average Tracker Time: " << util::accumulate(tracker_times.begin(),tracker_times.end(),0.0) / static_cast<float>(tracker_times.size())<< "ms\n";
    std::cout << "Max Tracker Time: " << *std::max_element(tracker_times.begin(),tracker_times.end()) << "ms\n";
    std::cout << "Min Tracker Time: " << *std::min_element(tracker_times.begin(),tracker_times.end()) << "ms\n";

    #endif

    return 0;
}


