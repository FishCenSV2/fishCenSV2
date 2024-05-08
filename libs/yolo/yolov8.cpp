#include "yolov8.hpp"
#include <fstream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

/** @brief Reads the engine file using an absolute file path.
*
* @param engine_file_path Absolute file path to .engine file.
* @return The raw binary of the engine file.
*/
std::vector<char> readEngineFile(const std::string& engine_file_path) {
    std::vector<char> engineData;
    std::ifstream engineFile(engine_file_path, std::ios::binary);
    if(!engineFile.good()) {
        std::cerr << "Error opening engine file: " << engine_file_path << std::endl;
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

/** @brief Reads the class .txt file using an absolute file path.
*
* @param class_file_path Absolute file path to .txt file.
* @return A vector containing the class names.
*/
std::vector<std::string> readClasses(const std::string& class_file_path) {
    std::vector<std::string> classes;
    std::ifstream infile(class_file_path, std::ios::in);

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
    //Some implementations might do this right after cudaMemCpy but that
    //means also doing cudaMalloc everytime as well. I think this should
    //be fine.
    cudaFree(_buffers[_input_index]);
    cudaFree(_buffers[_output_index]);
}

void YoloV8::init() {
    //Ignore any deprecation warnings.
    _runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(_logger));
    _engine = std::unique_ptr<nvinfer1::ICudaEngine>(_runtime->deserializeCudaEngine(_engine_data.data(), _engine_data.size(), nullptr));
    _context = std::unique_ptr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());

    //These names can be found by viewing the model with Netron.
    //Simply drag-and-drop the ONNX file (not the engine file) into it.
    _input_index = _engine->getBindingIndex("images");
    _output_index = _engine->getBindingIndex("output0");

    const auto idims = _engine->getBindingDimensions(_input_index);
    const auto odims = _engine->getBindingDimensions(_output_index);

    _max_out_dim = 0;

    input_len = idims.d[1] * idims.d[2] * idims.d[3];
    output_len = 1;

    //Loop over the output dimensions and find the max output dimension
    //as well as get the total number elements (output_len).
    for (auto &dim : odims.d) {
        if(dim < 0) {
            output_len *= 1;
        }

        else if (dim == 0) {
            break;
        }

        else {
            if(dim > _max_out_dim) {
                _max_out_dim = dim;
            }

            output_len *= dim;
        }
    }

    nvinfer1::Dims4 inputDims = {1, idims.d[1], idims.d[2], idims.d[3]};

    _context->setBindingDimensions(_input_index, inputDims);

    cudaMalloc(&_buffers[_input_index], input_len * sizeof(float));
    cudaMalloc(&_buffers[_output_index], output_len * sizeof(float));

    _output_data = std::make_unique<float[]>(output_len);

}

[[nodiscard]] cv::Mat YoloV8::preprocess(cv::Mat& image) {
    cv::Mat frame;

    //The padding is hardcoded based on an input of 640*480 so it will just
    //add 80 to the top and bottom of the the rows to make it 640*640
    cv::copyMakeBorder(image,frame, 80,80,0,0, CV_HAL_BORDER_CONSTANT, cv::Scalar(0,0,0));      
    //cv::copyMakeBorder(image,frame, 0,0,0,0, CV_HAL_BORDER_CONSTANT, cv::Scalar(0,0,0));

    /*
    TensorRT prefers NCHW but apparently OpenCV uses NHCW. We can use
    this function to also do some other pre-processing like scaling
    the input to be between 0 and 1 instead of 0 and 255.

    NOTE: The cv::Size(320,320) is also hardcoded based on the model.
          This should realistically be determined using idims.d 
    */

    //    return cv::dnn::blobFromImage(frame,1.0/255,cv::Size(320,320),cv::Scalar(0,0,0),true,false,CV_32F);
    //std::cout << frame.size() << std::endl;
    return cv::dnn::blobFromImage(frame,1.0/255,cv::Size(640,640),cv::Scalar(0,0,0),false,false,CV_32F);

}

void YoloV8::predict(cv::Mat& input) {
    cudaMemcpy(_buffers[_input_index], input.data, input.total() * input.elemSize(),cudaMemcpyHostToDevice);

    _context->executeV2(_buffers);

    cudaMemcpy(_output_data.get(), _buffers[_output_index], 
        output_len * sizeof(float),
        cudaMemcpyDeviceToHost);
        
}

[[nodiscard]] std::vector<Object> YoloV8::postprocess(float scale_factor) {

    constexpr int class_offset = 4; //The first four numbers are the xywh
    std::vector<Object> objects;    //Final output vector for bounding boxes.

    //The output is in dimensions of (# of classes + class_offset) * object_numbers
    cv::Mat data = cv::Mat(num_of_classes+class_offset,_max_out_dim,CV_32F,_output_data.get());

    for(int col = 0; col < _max_out_dim; col++) {
        float max = data.at<float>(4,col);
        int max_index = 0;

        //Find the row index in each column that has the highest number.
        //This is equivalent to the probability score.
        for (int i = 1; i < num_of_classes; i++) {
            if(max < data.at<float>(class_offset+i,col)) {
                max = data.at<float>(class_offset+i,col);
                max_index = i;
            }
        }

        if(max > _score_thresh) {
            
            float x = data.at<float>(0,col) * scale_factor;
            float y = data.at<float>(1,col) * scale_factor;
            float w = data.at<float>(2,col) * scale_factor;
            float h = data.at<float>(3,col) * scale_factor;

            float x_left_top = x - 0.5 * w;
            float y_left_top = y - 0.5 * h;
            
            _boxes.push_back(cv::Rect(x_left_top,y_left_top,w,h));
            _class_ids.push_back(max_index);
            _confidences.push_back(max);

            /*
            std::cout << max << " : [" <<
            x_left_top << ", " <<
            y_left_top << ", " <<
            w << ", " <<
            h << "]" << std::endl;
            //*/
        }
    }
    
    /*
    NMS (Non-max-suppression) to remove duplicate bounding boxes.
    The _indices vector will store all the indices that correspond to non-duplicate
    boxes.
    */
    cv::dnn::NMSBoxes(_boxes,_confidences,_score_thresh,_nms_thresh,_indices);

    //Loop over all indices to only get the non-duplicates.
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

    //std::cout << "# outputs from postprocess(inside): " << objects.size() << std::endl;
    
    /*for (int i = 0; i < objects.size(); i++) {
        std::cout << ">\t" << objects[i].rect.x << "x " << objects[i].rect.y << "y " << objects[i].rect.width << "w " << objects[i].rect.height << "h " <<
        "\t" << objects[i].label <<
        "\t" << objects[i].prob << std::endl;
    }
    */

    return objects;

}
