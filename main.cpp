#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        //suppress info-level messages
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

std::vector<char> readEngineFile(const std::string& engineFileName) {
    std::vector<char> engineData;
    std::ifstream engineFile(engineFileName, std::ios::binary);
    if(!engineFile.good()) {
        std::cerr << "Error opening engine file: " << engineFileName << std::endl;
        exit(EXIT_FAILURE);
    }

    engineFile.seekg(0,std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    engineData.resize(fileSize);
    engineFile.read(engineData.data(), fileSize);
    engineFile.close();

    return engineData;
}

std::vector<std::string> readClasses(const std::string& class_file) {
    std::vector<std::string> classes;
    std::ifstream infile(class_file, std::ios::in);

    while(infile.good()) {
        std::string data;
        std::getline(infile, data);
        if(data == "") {
            break;
        }

        classes.push_back(data);
        if (infile.eof()) {
            break;
        }
    }

    infile.close();

    return classes;
}


int main(){
    constexpr int image_dim = 320;
    constexpr int num_of_classes = 80;

    std::vector<std::string> classes = readClasses("/home/nvidia/Desktop/fishCenSV2/coco-classes.txt");
    std::vector<char> engine_data = readEngineFile("/home/nvidia/Desktop/fishCenSV2/yolov8n.engine");

    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr));
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    int32_t input_index = engine->getBindingIndex("images");
    int32_t output_index = engine->getBindingIndex("output0");

    auto idims = engine->getBindingDimensions(input_index);
    auto odims = engine->getBindingDimensions(output_index);

    size_t output_len = 1;

    for (auto &dim : odims.d) {
        if(dim < 0) {
            output_len *= 1;
        }

        else if (dim == 0) {
            break;
        }

        else {
            output_len *= dim;
        }
    }

    //1)Care should be taken here since a lot of the time 
    //most the dimensions change depending on the model.
    //2)Usually the zero dimension has a -ve value which
    //is probably representing some dynamic sizing thing. 
    nvinfer1::Dims4 inputDims = {1, idims.d[1], idims.d[2], idims.d[3]};
    // nvinfer1::Dims4 outputDims = {1, odims.d[1], odims.d[2], odims.d[3]};

    //Required.
    context->setBindingDimensions(input_index, inputDims);

    cv::Mat input_image = cv::imread("/home/nvidia/Desktop/fishCenSV2/images/two_obj.jpg");
    if (input_image.rows != image_dim && input_image.cols != image_dim) {
        cv::resize(input_image, input_image,cv::Size(image_dim,image_dim),cv::INTER_LINEAR);
    }
    cv::Mat img_cpy = input_image.clone();
    cv::cvtColor(input_image,input_image,cv::COLOR_BGR2RGB);
    input_image.convertTo(input_image,CV_32FC3);
    cv::normalize(input_image,input_image,0.0,1.0,cv::NORM_MINMAX);
   
    // cv::Mat rgb_channels[3];
    // cv::split(input_image,rgb_channels);

    // std::vector<cv::Mat> norm_channels;

    // //We normalize w.r.t how ImageNet does their normalization
    // norm_channels.push_back((rgb_channels[0] - 0.485) / 0.229);
    // norm_channels.push_back((rgb_channels[1] - 0.456) / 0.224);
    // norm_channels.push_back((rgb_channels[2] - 0.406) / 0.225);

    // cv::merge(norm_channels,input_image);

    //TensorRT prefers NCHW but apparently OpenCV uses NHCW.
    input_image = cv::dnn::blobFromImage(input_image);

    void* buffers[2];

    cudaMalloc(&buffers[input_index], input_image.total() * input_image.elemSize());
    cudaMalloc(&buffers[output_index], output_len * sizeof(float));

    cudaMemcpy(buffers[input_index], input_image.data,input_image.total() * input_image.elemSize(), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    context->executeV2(buffers);

    auto end = std::chrono::high_resolution_clock::now();

    auto output_data = std::make_unique<float[]>(output_len);
    cudaMemcpy(output_data.get(), buffers[output_index], 
                output_len * sizeof(float),
                cudaMemcpyDeviceToHost);

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms\n";

    //The below line does not create a copy but provides a nice
    //convenient way to access the flattened array.
    //NOTE:Hardcoded dimension should be removed.
    cv::Mat data = cv::Mat(84,2100,CV_32F,output_data.get());

    //Unless this is like the above which does no copying we should
    //change the loop below to interate without taking the transpose.
    cv::Mat data_trans = data.t();

    //Code from: 
    std::vector<cv::Rect> boxes, boxesNMS;
    std::vector<int> class_ids, class_idsNMS;
    std::vector<float> confidences;
    std::vector<int> indices;

    for(int row = 0; row < 2100; row++) {
        float max = data_trans.at<float>(row,4);
        int max_index = 0;
        for (int i = 1; i < num_of_classes; i++) {
            if(max < data_trans.at<float>(row,4+i)) {
                max = data_trans.at<float>(row,4+i);
                max_index = i;
            }
        }

        //Again hardcoded constant. This is the score threshold.
        if(max > 0.5) {
            float x = data_trans.at<float>(row,0);
            float y = data_trans.at<float>(row,1);
            float w = data_trans.at<float>(row,2);
            float h = data_trans.at<float>(row,3);

            float x_left_top = x - 0.5 * w;
            float y_left_top = y - 0.5 * h;
            
            boxes.push_back(cv::Rect(x_left_top,y_left_top,w,h));
            class_ids.push_back(max_index);
            confidences.push_back(max);
        }
    }

    //NMS to remove duplicate bounding boxes.
    //Hard coded constant is the NMS threshold.
    //Pretty sure this is also the IOU threshold.
    cv::dnn::NMSBoxes(boxes,confidences,0.5,0.8,indices);

    boxesNMS.resize(indices.size());
    class_idsNMS.resize(indices.size());
    for (int i = 0; i < indices.size(); i++) {
        int index = indices[i];
        boxesNMS[i] = boxes[index];
        class_idsNMS[i] = class_ids[index];
    }

    for (int i = 0; i < boxesNMS.size();i++) {
        cv::rectangle(img_cpy,boxesNMS[i],cv::Scalar(255,0,0),2);
        cv::putText(img_cpy,classes[class_idsNMS[i]],cv::Point(boxesNMS[i].x,boxesNMS[i].y-10),
                cv::FONT_HERSHEY_SIMPLEX,0.9,cv::Scalar(255,0,0),2);
    }

    cudaFree(buffers[input_index]);
    cudaFree(buffers[output_index]);

    std::cout << "Success!" << std::endl;

    cv::imshow("Output", img_cpy);
    cv::waitKey();
    
    return 0;
}
