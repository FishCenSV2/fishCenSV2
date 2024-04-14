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

/**
*   
* @brief Class that performs error logging for TensorRT 
*
*/
class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        //suppress info-level messages
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

/**
*
* @brief Class that represents the YOLOv8 model.
*
* An object which represents the YOLOv8 model. It works by
* loading the engine file for the model and using TensorRT
* to perform inference. The steps to run inference are as 
* follows:
*
* 1) Create YoloV8 object.
* 2) Initialize using class file and engine file.
* 3) Pre-process the image for the model.
* 4) Run inference on the pre-processed image.
* 5) Post-process the model's output and get a struct
*    containing the bounding boxes (including score, class id, etc.)
*
* NOTE: You may see a warning that says the following:
*
*       "Using an engine plan file across different models of devices is not recommended 
*        and is likely to affect performance or even cause errors."
*
*       This should be ignored since the engine file is built on this device.
*       Some googling shows that it is most likely a bug.
*
* @author Tristan Huen
*
*/
class YoloV8 {
    public:

        int num_of_classes;                ///< The number of classes the model identifies.
        size_t input_len;                  ///< Size of the input array (image size).
        size_t output_len;                 ///< Size of the output array from TensorRT.
        std::vector<std::string> classes;  ///< A vector of strings that contains the class names.

    private:
        static constexpr float _score_thresh = 0.5;  ///< Score threshold for a detection.
        static constexpr float _nms_thresh = 0.8;    ///< Threshold for NMS.
        int32_t _input_index;                        ///< Input index.
        int32_t _output_index;                       ///< Output index.
        int _max_out_dim;                            ///< The maximum dimension of the output.

        Logger _logger;                                        ///< Error logger.
        std::vector<char> _engine_data;                        ///< Engine data read from file
        std::unique_ptr<nvinfer1::IRuntime> _runtime;          ///< TensorRT runtime object
        std::unique_ptr<nvinfer1::ICudaEngine> _engine;        ///< TensorRT engine object
        std::unique_ptr<nvinfer1::IExecutionContext> _context; ///< TensorRT context object

        void* _buffers[2];                      ///< Buffers for input and output data.
        std::unique_ptr<float[]> _output_data;  ///< Output data from TensorRT
        std::vector<cv::Rect> _boxes;           ///< Vector of OpenCV Rects before NMS
        std::vector<int> _class_ids;            ///< Vector of class IDs before NMS
        std::vector<float> _confidences;        ///< Vector of confidence scores before NMS
        std::vector<int> _indices;              ///< Vector of indices for non-duplicate boxes (see postprocess function) 

    public:

        /** @brief Constructor for YoloV8 object
        *
        * @param classes_file_path Absolute file path to a .txt file containing the class names. 
        * @param engine_file_path Absolute file path to .engine file.
        * @return None
        */
        YoloV8(const std::string& classes_file_path, const std::string& engine_file_path);


        /** @brief Destructor for YoloV8 object.
        *
        */
        ~YoloV8();


        /** @brief Initializes the engine.
        *
        * @return None
        */
        void init();


        /** @brief Preprocesses the image that is to be passed to the model.
        *
        * @param image A cv::Mat of the image.
        * @return A preprocessed version of the input image.
        */
        cv::Mat preprocess(cv::Mat& image);

        /** @brief Performs inference on a pre-processed image.
        *
        * @param image A cv::Mat of the pre-processed image.
        * @return None
        */
        void predict(cv::Mat& input); 

        /** @brief Postprocesses the predictions from the model.
        *
        * @param scaling_factor The factor to scale the bounding boxes.
        * @return A vector of bounding box objects.
        */
        std::vector<Object> postprocess(float scale_factor);
  
};
