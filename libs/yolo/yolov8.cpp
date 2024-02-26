#include "yolov8.hpp"
#include <fstream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

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


YoloV8::YoloV8(const std::string& classes_file_path, 
               const std::string& engine_file_path) {

    classes = readClasses(classes_file_path);
    num_of_classes = classes.size();

    _engine_data = readEngineFile(engine_file_path);
}

YoloV8::~YoloV8() {
    cudaFree(_buffers[_input_index]);
    cudaFree(_buffers[_output_index]);
}

void YoloV8::init() {
    _runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(_logger));
    _engine = std::unique_ptr<nvinfer1::ICudaEngine>(_runtime->deserializeCudaEngine(_engine_data.data(), _engine_data.size(), nullptr));
    _context = std::unique_ptr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());

    _input_index = _engine->getBindingIndex("images");
    _output_index = _engine->getBindingIndex("output0");

    auto idims = _engine->getBindingDimensions(_input_index);
    auto odims = _engine->getBindingDimensions(_output_index);

    input_len = idims.d[1] * idims.d[2] * idims.d[3];
    output_len = 1;

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

    nvinfer1::Dims4 inputDims = {1, idims.d[1], idims.d[2], idims.d[3]};
    // nvinfer1::Dims4 outputDims = {1, odims.d[1], odims.d[2], odims.d[3]};

    //Required.
    _context->setBindingDimensions(_input_index, inputDims);

    cudaMalloc(&_buffers[_input_index], input_len * sizeof(float));
    cudaMalloc(&_buffers[_output_index], output_len * sizeof(float));

    _output_data = std::make_unique<float[]>(output_len);

}

cv::Mat YoloV8::preprocess(cv::Mat& image) {
    cv::Mat frame;

    cv::copyMakeBorder(image,frame, 80,80,0,0, CV_HAL_BORDER_CONSTANT, cv::Scalar(0,0,0));      

    //TensorRT prefers NCHW but apparently OpenCV uses NHCW.
    return cv::dnn::blobFromImage(frame,1.0/255,cv::Size(320,320),cv::Scalar(0,0,0),true,false,CV_32F);

}

void YoloV8::predict(cv::Mat& input) {
    cudaMemcpy(_buffers[_input_index], input.data, input.total() * input.elemSize(),cudaMemcpyHostToDevice);

    _context->executeV2(_buffers);

    cudaMemcpy(_output_data.get(), _buffers[_output_index], 
        output_len * sizeof(float),
        cudaMemcpyDeviceToHost);
        
}

std::vector<Object> YoloV8::postprocess(float scale_factor) {
    std::vector<Object> objects;

    cv::Mat data = cv::Mat(84,2100,CV_32F,_output_data.get());

    for(int col = 0; col < 2100; col++) {
        float max = data.at<float>(4,col);
        int max_index = 0;
        for (int i = 1; i < num_of_classes; i++) {
            if(max < data.at<float>(4+i,col)) {
                max = data.at<float>(4+i,col);
                max_index = i;
            }
        }

        //Again hardcoded constant. This is the score threshold.
        if(max > 0.5) {
            float x = data.at<float>(0,col) * scale_factor;
            float y = data.at<float>(1,col) * scale_factor;
            float w = data.at<float>(2,col) * scale_factor;
            float h = data.at<float>(3,col) * scale_factor;

            float x_left_top = x - 0.5 * w;
            float y_left_top = y - 0.5 * h;
            
            _boxes.push_back(cv::Rect(x_left_top,y_left_top,w,h));
            _class_ids.push_back(max_index);
            _confidences.push_back(max);
        }
    }

    //NMS to remove duplicate bounding boxes.
    //Hard coded constant is the NMS threshold.
    //Pretty sure this is also the IOU threshold.
    cv::dnn::NMSBoxes(_boxes,_confidences,0.5,0.8,_indices);

    for (int i = 0; i < _indices.size(); i++) {
        int index = _indices[i];
        objects.emplace_back();
        objects[i].rect = _boxes[index];
        objects[i].label = _class_ids[index];
        objects[i].prob = _confidences[index];
    }
    
    _indices.clear();
    _boxes.clear();
    _confidences.clear();
    _class_ids.clear();

    return objects;

}
