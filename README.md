# fishCenSV2
A fish-counting and identifier system using machine learning that can be used to track salmon as they swim upstream creeks.

## Table of Contents

- [fishCenSV2](#fishcensv2)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Machine Learning](#machine-learning)
    - [YOLOv8](#yolov8)
    - [ByteTrack](#bytetrack)
    - [TensorRT](#tensorrt)
  - [How the Code Works](#how-the-code-works)
  - [The Code](#the-code)
    - [Server Thread](#server-thread)
    - [UDP Thread](#udp-thread)
    - [Main Loop Thread](#main-loop-thread)
  - [Appendix](#appendix)
    - [Constexpr](#constexpr)
    - [Alias Declarations](#alias-declarations)
    - [Noexcept](#noexcept)
    - [Lambda Expressions](#lambda-expressions)
    - [Threading](#threading)
    - [RAII](#raii)
    - [Mutex Locking with RAII](#mutex-locking-with-raii)
    - [Smart Pointers (Unique and Shared)](#smart-pointers-unique-and-shared)
    - [Condition Variable](#condition-variable)
    - [Move Semantics](#move-semantics)
    - [Return Value Optimization (RVO)](#return-value-optimization-rvo)


## Introduction
FishCenSV2 is a C++ project that uses machine learning to classify and count salmon as they swim. It draws bounding boxes on the fish, tracks their movement across the camera, and counts them once they cross the middle of the camera. All of this runs on a NVIDIA Jetson TX2 which performs inference using YOLOv8 and tracking using ByteTrack. Video feed is grabbed from a Raspberry Pi 4 over UDP.

The first version of the FishCenS project only used a Raspberry Pi 4 and a Google Coral Accelerator to help with the machine learning. Additionally, it was limited to TensorFlow Lite models and OpenCV trackers. We aim to build upon the previous project by improving the algorithms and hardware used.

This repository contains all the code for the Jetson and documentation for everything. The next few sections cover a more in-depth breakdown of all of the code in the `main.cpp` file. More detailed explanations about the classes/libraries used can be found in the [`libs`](https://github.com/FishCenSV2/fishCenSV2/tree/main/libs) folder. Each folder in there corresponds to a "library" which has documentation as well. Some of it may seem more exhaustive than necessary but since not everyone knows modern C++ or machine learning I feel it is needed.

It is highly recommended to read the [Appendix](#appendix) which contains C++ features that may need more clarification.

**NOTE: As of right now everything is still WIP. Many things are missing and the explanations may not be the best.**

## Machine Learning
I thought I would put this here in order to explain some of the machine learning libraries, tools, and how we actually set everything up.

### YOLOv8

YOLOv8 is a machine learning real-time object detection model. It simply takes in an image and outputs bounding boxes around the objects it detects in the image. It also can differentiate between two different types of objects such as a dog or a cat.

We train the YOLOv8 model using weights from a previously trained model. This is often better than training the model from scratch and also requires less training data. The model is trained using Google Colab as it offers power cloud computing GPUs like the NVIDIA A100. Once the model is trained we have a file containing all the weights and everything we need to run the model on some inputs. An additional step is the conversion to another format called ONNX. This is a more universal format as machine learning models can be trained using PyTorch, or TensorFlow which do not use the same format. We use ONNX for our project. 

### ByteTrack 
ByteTrack is an multi-object tracker that attempts to keep track of bounding boxes throughout successive video frames by assigning unique IDs to them. Essentially it relies on a detector, like YOLOv8, to create the bounding boxes for each video frame. The way it works is as follows
- Run the detector on the current video frame to get bounding boxes
- Tracker assigns unique IDs to these bounding boxes
- Run the detector on the next video frame to get new bounding boxes
- Tracker takes in these new bounding boxes and tries to associate them with the ones from the previous frame. 
- If it succeeds then the ID's location changes to the new bounding box's location. Else a new ID is created for a new bounding box.

Note that after some period of time if an ID can't be successfully retracked then it will be deleted.

ByteTrack is very good due to its speed and the ability to deal with occlusion. Occlusion is when an object is unable to be seen for several frames due to being blocked by something. If this happens for a short-enough period of time then ByteTrack is able to retrack the object.

### TensorRT
TensorRT is a NVIDIA library that speeds up inference time for AI models. Inference refers to the AI model taking in an input and spitting out a output. Much of the details are in the YOLOv8 library code but in short TensorRT converts a machine learning model (in ONNX format) into an engine file that is optimized for the architecture of the computer.  It can then use this engine file to run the model.  For our use we use a command line tool called `trtexec` to create the engine file, and use the C++ library to run the engine for inference. More info about `trtexec` can be found [here](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec).

NOTE: I will add more info here later. It can be finnicky to deal with.

## How the Code Works
The code has three parallel running processes: two servers, and a main loop. The first server is a UDP stream that just sends an annotated video feed to a client. Annotated meaning bounding boxes, live count, etc. The second server sends the actual counting data to the client. Both of these run independently of a main loop function which roughly does the following
- 1) Read a video frame from the Raspberry Pi's UDP stream.
- 2) Pre-process the video frame and send it to the machine learning detection model.
- 3) Read the output of detections directly from the detection model.
- 4) Post-process the output and update the tracker with it.
- 5) Get a list of unique IDs for each detection from the tracker and use that to determine if any detected object has crossed the middle of the screen.
- 6) Repeat from step 1

## The Code
This section dives into the running of the `main.cpp` file. There some added comments that don't exist in the actual code. Let's start by looking at some global variables that must be mentioned.

```cpp
bool end_program_flag = false;      //Not atomic since it does not mix well with cond_var.
std::condition_variable cond_var;   //Condition variable for UDP queue.
std::mutex m_udp;                   //Mutex for UDP queue
std::mutex m_data;                  //Mutex for counting data;
std::queue<cv::Mat> frame_queue;    //Queue of frames for UDP stream.
```
This is mostly self-explanatory. The UDP server relies on a queue of video frames which the main loop code pushes onto it. Thus, we have a producer-consumer problem which a condition variable can help with. See the [Appendix](#appendix) for more info about condition variables. Next let's take a look at the main function

```cpp
int main() {
  boost::asio::io_context io_context;
  Server server(io_context, 1234, m_data);

  //Launch the threads. Execution starts immediately
  std::thread server_th([&]() {server.run();});
  std::thread udp_stream_th(udp_stream);
  std::thread main_loop_th(main_loop, std::ref(server));

  //Block until threads are done running. Prevent main from returning early.
  server_th.join();
  udp_stream_th.join();
  main_loop_th.join();
  
  return 0;
}
```

The first two lines setup the server that sends counting data to clients. Note that the global variable mutex `m_data` is passed into its constructor. The next step is the creation of three threads. The `server_th` thread uses a lambda function (see [Appendix](#appendix)) since it is only one line of code.  The next thread `udp_stream_th` calls the `udp_stream` function. This is a live video feed of the camera with bounding boxes and counts. The final thread `main_loop_th` calls the `main_loop` function. This is the function that does inference, tracking, and other things. Finally `.join()` is called on each thread (again see [Appendix](#appendix))

### Server Thread
The server thread uses a lambda function which only executes the following
```cpp
server.run();
```
This is a blocking function which is why it gets its own thread. A `server.stop()` is present in the `main_loop` function to stop the server when the user wishes to exit.

### UDP Thread
The `udp_stream` function starts with the following
```cpp
const std::string ip = "239.255.0.1"; //Multicast address.
constexpr unsigned port = 12345;
const auto udp_ip = asio::ip::address::from_string(ip);
constexpr int compression_factor = 80;
std::vector<uint8_t> buff;
const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY,compression_factor};

```
For more info on `constexpr` see the [Appendix](#appendix).  The first three lines just setup some UDP variables. The next three lines setup JPEG compression that will be applied to each video frame. The `buff` vector will contain the pure bytes of the compressed image that will be sent over the stream.

The next couple lines in the code are straightforward and start the UDP server.
```cpp
asio::io_service io_service;
asio::ip::udp::socket socket(io_service);
asio::ip::udp::endpoint receiver_endpoint(udp_ip,port);
system::error_code ec;

socket.open(asio::ip::udp::v4());
```

Next we can look at the final piece which is the `while` loop
```cpp
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
```

This is similar to the condition variable code in the appendix. Once `cond_var` is notified we check if the `end_program_flag` is set and if it is then we end the loop and our function returns which ends the thread. Otherwise, we get an image frame from the queue, unlock the lock, encode the image in JPEG format, and send over the UDP socket to the client.

### Main Loop Thread
The main loop function contains all the inference, and tracking code. First we take a look at the following lines
```cpp
//For some reason relative paths don't work
const std::string classes_file_path = "/home/nvidia/Desktop/fishCenSV2/coco-classes.txt";
const std::string engine_file_path = "/home/nvidia/Desktop/fishCenSV2/yolov8n.engine";

//Address is the Jetson's. This should always be the same address as configured by the router.
const std::string pipeline = "udpsrc address=192.168.0.103 port=5000 ! application/x-rtp, payload=96, encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=480 ! videoconvert ! video/x-raw, format=BGR, width=640, height=480 ! appsink";
```

**IMPORTANT!** Change the filepaths if you are using different models. The `.txt` file should have each class separated line-by-line and be in the same order as the model (check Netron for order). For example, line one is index 0. You can see this in the `MODEL PROPERTIES` panel under `METADATA` and `names`.

The first two lines are for intializing the machine learning model and we can ignore it for now. The last line is fed into the `.open()` function of the `cv::VideoCapture` class. Notably, this is more complicated than a regular UDP url since we are using GStreamer as the backend for video instead of FFMPEG (it didn't work). Not much else is to be said as much of this was hacked together from examples online. 

Let us take a look at the next lines
```cpp
constexpr int fps = 50; //FPS that tracker uses

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

//Construct the machine learning model with the two files from before
YoloV8 detector(classes_file_path,engine_file_path);

std::cout << "Initializing YoloV8 engine\n";

detector.init();
```

The first line sets the FPS for the tracking algorithm (initialization not seen here). The next couple lines are the left and right counts for the species. Left and right mean the direction of travel when crossing the middle of the screen At the moment this just counts cellphones and anything else that is not a cellphone.  The lines after that are just more setup and some will be brought up again later. Note that the second-to-last line constructs the machine learning detector using the two strings mentioned at the start.

NOTE: PLEASE look at the documentation for the yolo class [here](https://github.com/FishCenSV2/fishCenSV2/tree/main/libs/yolo). There is the possiblity that something may break due to using a different model.

Next we have the following code
```cpp
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
```

The `NO_PI` is used for simple webcam testing. We can see that for our normal operation we pass the GStreamer pipeline string from earlier into the the `.open()` function. 

Next we move onto the `while` loop. Here I removed most of the timer code from the original since that just measures execution times.

```cpp
while(1) {

  //Read frame from UDP stream
  cap.read(frame);
  
  if(frame.empty()) {
    std::cerr << "ERROR! Blank frame grabbed\n";
    break;
  }
    
  //Pre-process the video frame and assign it to a cv::Mat.
  //This pre-processing is described more in the actual library documentation.
  //For now you can simply think of it as adding padding to the original 640x480
  //to give us 640x640, and a further resizing to 320x320.
  cv::Mat input = detector.preprocess(frame);
  
  //Run the model detection on the cv::Mat
  detector.predict(input);
  
  //Post-process the model's detections to get a vector of bounding boxes
  //The "2" is a multiplicative scaling factor since our model only takes in 
  //320x320 images while the live feed is 640x640. Essentially we scale the
  //bounding boxes to the proper dimensions.
  objects = detector.postprocess(2);
  
  //Add padding to the original frame that the user will see.
  //We use this instead of the pre-processed one due to some changes made by the function.
  cv::copyMakeBorder(frame,frame, 80,80,0,0, CV_HAL_BORDER_CONSTANT, cv::Scalar(0,0,0));

  //Other code below this....

}
```

This code is how steps 1) to 4) in the "How the Code Works" section work.  The detector's `.predict()` doesn't actually return the detections because we must find them from the raw output of TensorRT engine. Furthermore, I found it more useful to make `predict()` return nothing since we must do post-processing anyways. The post-processing returns a vector of bounding boxes which we assign to the variable `objects`. Recall that we previously had this variable declared as

```cpp
std::vector<Object> objects;  //Vector of bounding boxes
```
The `Objects` type is a struct which comes from the ByteTrack library. 
```cpp
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
```
Where `rect` is an OpenCV rectangle, `label` is the class/species that the object belongs to, and `prob` is the classification score.

The next portion of code is the tracking and counting
```cpp
//We are still in the while(1) loop.

output_stracks = tracker.update(objects);

if(output_stracks.size() > 0) {

  //Loop over all output tracks and check if they crossed the boundary.
  for(int i = 0; i < output_stracks.size(); i++) {
  
    //Get the current position of the object
    int current_position = static_cast<int>(.5 *(output_stracks[i].tlbr[0] + output_stracks[i].tlbr[2]));
  
    //If track_id of the object isn't in the map then add it and its respective 
    //current position.
    if(previous_pos.count(output_stracks[i].track_id) == 0) {
      previous_pos[output_stracks[i].track_id] = current_position;
      continue;
    }

    //Else get the previous position of the object and find out if it has 
    //crossed the middle line using its current position.
    else {
      int previous_position = previous_pos[output_stracks[i].track_id];
  
      if(current_position - previous_position > 0 && previous_position < 320 && current_position >= 320) { {
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

      //Update the previous position to be the current position for the next video frame
      previous_pos[output_stracks[i].track_id] = current_position;
    }
  
    //Code below directly grabbed from one of the ByteTrack TensorRT files
    std::vector<float> tlwh = output_stracks[i].tlwh;
    float prob_score = objects[i].prob * 100;
    
    //Draw bounding boxes, and object classes to frame.
    cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
    cv::putText(frame, cv::format("%s#%d", (detector.classes[objects[i].label]).c_str(),output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1]-24),
    0, 0.6, cv::Scalar(0,0,255),1,cv::LINE_AA);
    cv::putText(frame, cv::format("%.2f%s",prob_score,"%"), cv::Point(tlwh[0], tlwh[1]-5),
    0, 0.6, cv::Scalar(0,0,255),1,cv::LINE_AA);
    cv::rectangle(frame, cv::Rect(tlwh[0],tlwh[1],tlwh[2],tlwh[3]),s,2);
  
  }
  
  {
    //Update the server's counting data with the new counts.
    std::lock_guard<std::mutex> l(m_data);
    server.data = {l_cell_count, r_cell_count, left_count, right_count};
  }

}

//More code below this...

```
Recall that we declared the following variables
```cpp
std::vector<STrack> output_stracks;         //Vector of tracked objects
std::unordered_map<int, int> previous_pos;  //Key = Track ID, Val = Previous Position
```

The tracker takes in the post-processed bounding boxes and returns a vector of `STrack` objects which we assign to `output_stracks`. These `STrack` objects contain the coordinates of the bounding boxes and a unique ID. Let's focus on one box for now. We use the bounding box's coordinates to get the current position of the box. We then use the `previous_pos` map (equivalent to the Python dictionary) to check if the unique ID is contained within it. If it's not we add the unique ID as a new key whose value is the current position we had just calculated. If the unique ID is in the map then we use the calculated current position and the previous position associated with that unique ID from the `previous_pos` map to see if the object has crossed the middle line. There's also some other logic for figuring out the direction of crossing but that is simple enough to understand. Finally we update the previous position in the map with the current position. 

The next piece of code removes IDs that are no longer being tracked from the `previous_pos` map.
```cpp
//We are still in the while(1) loop.

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
```

We will skip some lines of code since they are just for drawing things to the video frame. The final lines of code in the while loop are the following
```cpp
//We are still in the while(1) loop.

//Drawing code above this...

{
    std::lock_guard<std::mutex> l{m_udp};
    frame_queue.push(frame);
}

//Notify UDP thread
cond_var.notify_one();

if(cv::waitKey(1) >= 0) {
    std::lock_guard<std::mutex> lock(m_udp);
    end_program_flag = true;
    cond_var.notify_one();
    server.stop();
    break;
}

//The while(1) ends here and so does main_loop()
```
After doing all the drawing to the frame we push it onto the UDP frame queue and notify the condition variable. Finally, we check if the user hits any key in the OpenCV window to end the program. This covers it for the `main.cpp` file.

## Appendix
This appendix is not meant to be a complete tutorial on each of these C++ features but a good enough description on them. If you would like to read more on them then I highly recommend Scott Meyers' Effective Modern C++.

### Constexpr
---
The keyword `constexpr` defines a value that can be evaluated at compile time. It is similar to a `#define` but more powerful. 

```cpp
constexpr float a = 3.0;
constexpr float b = 6.0;
constexpr float c = a * b + b * 2;
```

All of these expressions will be evaluated at compile time. Of course this is only limited to things that you know can be done at compile time instead of run time. For example, a vector before C++20 can not be `constexpr` since it is dynamically allocated. For our purposes `constexpr` doesn't have a lot of use but it can be used to make compile time lookup tables. This is possible since we can also have `constexpr` functions.

### Alias Declarations
---
C++ allows the use of a type alias which allows us to create a new name for some type. For example,

```cpp
using uint = unsigned;
using time_point = std::chrono::high_resolution_clock::time_point;

uint n = 3;
time_point start = std::chrono::high_resolution_clock::now();
```

The other way would be to use `typedef` but the syntax can be more confusing
```cpp
//Function pointers example

//C++11 and above
using func = void(*)(int);

//C++03
typdef void (*func)(int);

```
You can read more about aliases and typdefs [here](https://learn.microsoft.com/en-us/cpp/cpp/aliases-and-typedefs-cpp?view=msvc-170)

### Noexcept
---
A function that is declared with `noexcept` means the function is guaranteed to never throw an exception. For example, all destructors and memory deallocation functions are declared `noexcept` implicitly. Additionally, `noexcept` allows the compiler to generate better object code but the compiler may not do this so don't go throwing it on all of your functions. An example of the syntax can be seen below

```cpp
int f(int x) noexcept; 
```

### Lambda Expressions
---
A lambda expression is convenient way for creating function objects that can be passed as arguments to other functions or executed right where it is declared. Consider the following example

```cpp
auto func = [](int a, int b) {
   return a * b;
};

int b = func(3,4);

```
Here the `[]` is called the capture clause. It allows us to specify whether outside variables used in the lambda's body should be captured by reference or by value in the enclosing scope. Enclosing scope meaning the scope that the lambda expression is contained in. Here is a list of potential capture clauses,

- `[]`: No access to any variables outside lambda within enclosing scope.
- `[&]`: All variables in the enclosing scope are captured by reference.
- `[=]`: All variables in the enclosing scope are captured by value.
- `[&var1, var2]`: Variable `var1` is captured by reference, while `var2` by value.
- `[&, var2]`: Variable `var2` is captured by value. All others by reference.
- `[=, &var2]`: Variable `var2` is captured by reference. All others by value.

For example,
```cpp
int a = 3;
int b = 2;

//Captures variables `a` and `b` by value. An empty clause [] causes a compile error here.
auto func = [=](int d, int c) {
   return a * b * d * c;
};

//Returns 3 * 2 * 1 * 4
int m = func(1,4);
```



### Threading
---
C++11 introduced multithreading which allows concurrent execution to be possible. Let's look at a simple example

```cpp
#include <thread>
#include <iostream>
#include <chrono>

void long_work(int time) {
  std::this_thread::sleep_for(std::chrono::seconds(time));
  std::cout << "Finished my long work!\n";
}

void short_work(int time) {
  std::this_thread::sleep_for(std::chrono::seconds(time));
  std::cout << "Finished my short work!\n";
}

int main() {
  int long_time = 5;
  int short_time = 1;

  std::thread t1(long_work, long_time);
  std::thread t2(short_work, short_time);

  t1.join();
  t2.join();

  return 0;
}
```

A thread is constructed by passing the function name and the corresponding arguments to the function. When a thread is constructed it begins executing the function supplied to it. In our case `t1` and `t2` both begin running `long_work` and `short_work`. This is all happening concurrently alongside main. Right afterwards we call `.join()` on both of these threads. `.join()` blocks execution until the thread's function returns. So in our case `t1.join()` will block until `t1` is finished, and the same applies to `t2`. We must always call `.join()` (or `.detach()` but ignore that one) in order to prevent the main function from exiting before the threads finish. Otherwise this will cause a runtime exception to be thrown. Once both threads are finished running then our main function returns. 

A small note to make is that launching a thread with a function that takes an argument by reference must use `std::ref`

```cpp
#include <thread>
#include <vector>

void func(std::vector<int>& nums, int param) {
  //Some code here...
}

int main() {
  std::vector<int> nums = {1,2,3,4};
  int param = 6;

  std::thread t1(func, std::ref(nums),param);

  t1.join();

  return 0;
}
```

### RAII
---
RAII stands for "Resource acquistion is initialization" but this name is honestly pretty bad. A better name is "Scoped-based resource management". Simply put, an object's lifetime is controlled by its scope and once it goes out of scope the object is destroyed and the resource is freed. We can see why this matters with the example of a mutex.

```cpp
std::mutex mtx;

void func() {
	mtx.lock();
	//Some code here...
	mtx.unlock();
}
```

In this case if the code in between locking and unlocking the mutex throws an exception then the mutex is never unlocked and we run into deadlock. Of course, you could write a try-catch block which would also involve unlocking the mutex in the catch portion but that is messy. Instead we can use an RAII-style locking mechanism with `std::lock_guard`

```cpp
std::mutex mtx;

void func() {
	std::lock_guard<std::mutex> lock{mtx};
	//Some code here...
	//Mutex is unlocked when `lock` goes out of scope.
}
```

In this case the mutex is automatically locked when we construct the `std::lock_guard` object and if an exception is thrown or the function returns, the object goes out of scope, its destructor is called, and the mutex will be automatically unlocked.  

RAII is also used for dynamic memory allocation which is covered in the Smart Pointers section.

### Mutex Locking with RAII
I thought I would create a brief section that talked about mutexes. We already know from the section on RAII that manually locking and unlocking the mutex is not safe which is where the RAII wrapper class `std::lock_guard` came into play. There are at least two other RAII mutex locking classes which are `std::unique_lock` and `std::scoped_lock`.

The class `std::unique_lock` is like `std::lock_guard` with the added bonus of being able to manually unlock and lock the mutex. `std::lock_guard` is a scoped based lock which can only unlock the mutex when the lock goes out of scope. `std::unique_lock` will still do the same automatic locking upon creation and unlocking upon destruction but now you can control the locking/unlocking manually. It also provides some other features but they aren't relevant for our purpose.

The class `std::scoped_lock` can be considered a superior version of `std::lock_guard` as it can lock multiple mutexes all at once and uses a deadlock avoidance algorithm. Of course the code we have currently uses `std::lock_guard` everywhere but at the moment we are not acquiring several mutexes so we can stay with it for now (it can always be changed with a CTRL-F). Another thing to note is that `std::scoped_lock` can cause some issues due to user error. See the following code where a user may forget to initialize the lock with a mutex

```cpp
//From this thread: https://stackoverflow.com/questions/43019598/stdlock-guard-or-stdscoped-lock

std::scoped_lock lock; //A runtime error
std::lock_guard lock;  //A compile time error
```

Resolving errors at compile time is no doubt better than resolving errors at run time.

### Smart Pointers (Unique and Shared)
---
In C++ dynamic memory allocation can be done with the operators `new` and `delete`. However, this is a similar situation to the mutexes `lock` and `unlock` problem where `unlock` can be never called due to an exeception. In this case if `delete` is never called then the memory we allocated is never released. Just like with mutexes RAII comes to the rescue in the form of smart pointers. Smart pointers take care of allocation and deallocation upon destruction without the need to call `new` or `delete`. We will look at two smart pointers `std::unique_ptr` and `std::shared_ptr`.

Unique pointers have exclusive ownership of the object they point to and take care of their destruction. Thus, we can't copy unique pointers since they would both try to destroy the object. However, we can move them (see section on [Move Semantics](#move-semantics)). A simple example is shown below

```cpp
#include <iostream>
#include <memory>

int main() {

    //Create unique pointer to dynamic float array. It owns this array.
    std::unique_ptr<float[]> output = std::make_unique<float[]>(5);

    //Prints elements of array. Will print "Array at [0] = 0" and "Array at [1] = 0"
    std::cout << "Array at [0] = " << *(output.get()) << "\n";
    std::cout << "Array at [1] = " << *(output.get()+1) << "\n";

    //Access array using raw pointer returned by .get() of unique pointer
    *output.get() = 1;
    *(output.get()+1) = 2;

    //Prints elements of array. Will print "Array at [0] = 1" and "Array at [1] = 2"
    std::cout << "Array at [0] = " << *(output.get()) << "\n";
    std::cout << "Array at [1] = " << *(output.get()+1) << "\n";

    //Memory of array deallocated when `output` goes out of scope
    return 0;
}

```
Unique pointers can also be passed a custom deleter functions that is invoked when the resource is released but we won't likely need one. 

A shared pointer is used for shared-resource ownership. Multiple pointers all point to an object and will ensure its destruction. When the last pointer to the object is destroyed or points to something else then it will destroy the object. Shared pointers can tell if they are the last pointer by looking at the resource's reference count which keeps track of the number of pointers pointing to it. Shared pointers have more overhead and complications than the unique pointer which should be kept in mind. We only use a shared pointer in one piece of our code (the server) so I won't go through all the details. A simple example is shown below

```cpp
#include <iostream>
#include <memory>

class MyClass {
public:
    MyClass(int val) : value(val) {
        std::cout << "Constructor called. Value: " << value << std::endl;
    }
    
    ~MyClass() {
        std::cout << "Destructor called. Value: " << value << std::endl;
    }
    
    void setValue(int val) {
        value = val;
    }
    
    int getValue() const {
        return value;
    }

private:
    int value;
};

int main() {
    //Creating shared pointers using make_shared.
    std::shared_ptr<MyClass> ptr1 = std::make_shared<MyClass>(5);
    std::shared_ptr<MyClass> ptr2 = ptr1; // Shared ownership
    
    //Accessing object through shared pointers.
    std::cout << "Value of ptr1: " << ptr1->getValue() << std::endl;
    std::cout << "Value of ptr2: " << ptr2->getValue() << std::endl;
    
    //Modifying object through one of the shared pointers.
    ptr1->setValue(10);
    
    std::cout << "Value of ptr1 after modification: " << ptr1->getValue() << std::endl;
    std::cout << "Value of ptr2 after modification: " << ptr2->getValue() << std::endl;
    
    //Resetting one of the shared pointers.
    //This makes ptr1 point to nullptr and decrements the reference count.
    ptr1.reset();
    
    std::cout << "ptr1 reset. Value of ptr2: " << ptr2->getValue() << std::endl;
    
    //ptr2 is still holding the object, so it will be destructed when ptr2 is reset or goes out of scope.
    return 0;
```

The output is the following
```
Constructor called. Value: 5
Value of ptr1: 5
Value of ptr2: 5
Value of ptr1 after modification: 10
Value of ptr2 after modification: 10
ptr1 reset. Value of ptr2: 10
Destructor called. Value: 10
```

In essence, shared pointers are like the C++ way of garbage collection.

### Condition Variable
---
Consider the problem of a producer and consumer. The producer makes food at a slow pace while the consumer plates the food at a faster pace and then eats it. We can imagine this in a programming way as two threads (producer and consumer) which share access to queue of objects (food). The question is how do we synchronzize both of these threads so that the consumer doesn't try pop an item off of the queue if it is empty? In other words how do we prevent the consumer from plating food before the producer is done? This is where condition variables come into play. 

```cpp
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <string>

Struct Food {
  std::string name;
}

std::condition_variable cond_var;
std::mutex m;
std::queue<Food> food_queue;

void produce_food() {
  while(1) {
      Food food = make_food(); //Imagine this takes a long time.

      {
        std::lock_guard<std::mutex> l{m};
        food_queue.push(food);
      }

      cond_var.notify_one();
  }
}

void consume_food() {
  while(1) {
    std::unique_lock<std::mutex> lock{m};
    cond_var.wait(lock, [](){ return !food_queue.empty(); });

    //Equivalent to plating the food
    Food food = food_queue.front();
    food_queue.pop();
    lock.unlock();

    eat_food(Food); 
  }
}

int main() {
  std::thread t1(produce_food);
  std::thread t2(consume_food);

  t1.join();
  t2.join();

  return 0;
}
```

Let's start with the `produce_food` function.

```cpp
void produce_food() {
  while(1) {
      Food food = make_food(); //Imagine this takes long

      {
        std::lock_guard<std::mutex> l{m};
        food_queue.push(food);
      }

      cond_var.notify_one();
  }
}
```

Here we "make the food", lock the mutex, push the food onto the queue, unlock the mutex, and then notify the condition variable `cond_var` to wakeup. Let's now take a look at the `consume_food` function.

```cpp
void consume_food() {
  while(1) {
    std::unique_lock<std::mutex> lock{m};
    cond_var.wait(lock, [](){ return !food_queue.empty(); });

    //Equivalent to plating the food
    Food food = food_queue.front();
    food_queue.pop();
    lock.unlock();

    eat_food(Food); 
  }
}
```

The first thing we do is create a unique lock and lock the mutex. Condition variables only work with unique locks. Next we call `.wait()` on `cond_var` which, if it has not been notified, unlocks the mutex and suspends the execution of the thread. If it has been notified then it will check that the lambda function evaluates to true and continue with the rest of the code (plate the food and unlock the mutex). If the thread was suspended and `cond_var` gets notified then the thread will continue executing by relocking the mutex and continuing with the previous process we described. 

A problem that can arise with condition variables is something called a spurious wakeup. This happens when the thread waiting for the condition variable manages to wakeup up for no reason even if it had not been notified. This could cause the consuming thread to execute before the producer thread is finished. However, our use of the lambda function (called the predicate in this case) prevents the thread from continuing since the queue is only non-empty when our producer is finished. Predicates should always be used. 

This section on conditon variables is derived from a great post that can be found [here](https://chrizog.com/cpp-thread-synchronization)

### Move Semantics
---
**This explanation of move semantics is not going to be technically accurate or in-depth since we do not utilize move semantics that much. Much of this section is based on chapter 5 of Scott Meyers' Effective Modern C++** 

In C++ we have the concept of lvalues and rvalues. An lvalue is something that has a name or memory address. An rvalue is something that has no memory location and can be things like temporary objects or literal constants (e.g. 420, 3, etc.). Take a look at the following example
```cpp
int x = 3;
std::string str = std::string("HI");
```
The variables `x` and `str` are lvalues. Meanwhile `3` and `std::string("HI")` are rvalues. Here `std::string("HI")` returns a temporary string object that must be assigned to a variable. Historically the l in lvalue and r in rvalue referred to which side the value would appear on assignment operator (=). In other words l means left, and r means right. 

Move semantics is all about "moving" resources to avoid unecessary copy operations. To understand this further let's declare two vectors of some class `Object`

```cpp
std::vector<Object> objects = {obj1, obj2, obj3};
std:vector<Object> objects_new; 
```

If we directly do `objects_new = objects;` then all of the items in `objects` are copied directly to `objects_new`. However, in some cases a copy could be expensive (long execution time) and we don't care about the `objects` vector. In that case move semantics allows us to move all the items from `objects` to `objects_new` without copying them. This leaves the `objects` in a valid but unspecified state. For the vector class a "valid but unspecified state" is just a empty vector. The syntax to do this all of this is as follows

```cpp
//For a vector it is simply like or equivalent to a pointer swap.
objects_new = std::move(objects);
```

Now `std::move` actually doesn't do any "moving". What it actually does is it casts its argument to an rvalue. An rvalue is a candidate for moving so essentially `std::move` just tells the compiler that the object may be moved from. However, rvalues aren't always candidates for moving so keep that in mind. Additionally, not all types support move operations or they may be costly. For example, `std::mutex` is neither moveable or copyable. Additionally, applying a move on a `std::array` takes the exact same time complexity as copying due to underlying implementation details (`std::array` is not similar to `std::vector` and other heap storing containers). 

### Return Value Optimization (RVO)
---

Consider this simple function that returns a vector
```cpp
std::vector<int> createVector(int a, int b, int c) {
   std::vector<int> v= {a,b,c};
   return v;
}
```

And we can call this function to create a vector

```cpp
std::vector<int> myvector = createVector(3,4,5);
```

One might think that when we call `createVector` it will construct a vector `v` with the three arguments, copy all of the elements to `myvector`, and destroy `v` once the function ends. So, using move semantics you might think we can avoid the copying and do this instead

```cpp
std::vector<int> createVector(int a, int b, int c) {
   std::vector<int> v= {a,b,c};
   return std::move(v); //NO
}
```
But this is not needed at all! The compiler will actually perform copy elision to avoid this unecessary copy. This is called return value optimization (RVO). So don't use `std::move` with the return statement and trust the compiler!
