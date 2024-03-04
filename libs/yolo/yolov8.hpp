#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <BYTETracker.h> //Needed for Object struct.


class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        //suppress info-level messages
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class YoloV8 {
    public:

        int num_of_classes;
        size_t input_len;
        size_t output_len;
        std::vector<std::string> classes;

    private:
        static constexpr float _score_thresh = 0.5;
        static constexpr float _nms_thresh = 0.8;
        int32_t _input_index;
        int32_t _output_index;
        int _max_out_dim;

        Logger _logger;
        std::vector<char> _engine_data;
        std::unique_ptr<nvinfer1::IRuntime> _runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> _engine;
        std::unique_ptr<nvinfer1::IExecutionContext> _context;

        void* _buffers[2];
        std::unique_ptr<float[]> _output_data;
        std::vector<cv::Rect> _boxes;
        std::vector<int> _class_ids;
        std::vector<float> _confidences;
        std::vector<int> _indices;

    public:

        YoloV8(const std::string& classes_file_path, const std::string& engine_file_path);
        ~YoloV8();

        void init();
        cv::Mat preprocess(cv::Mat& image);
        void predict(cv::Mat& input); 
        std::vector<Object> postprocess(float scale_factor);
  
};
