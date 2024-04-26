# YoloV8
This documentation contains breakdowns of some of (not all) of the functions used in the `YoloV8` class. Much of this is based the TensorRT documentation found [here](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/quick-start-guide/index.html#run-engine-c) and also [here](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/api/c_api/classnvinfer1_1_1_i_execution_context.html)

**NOTE: This documentation does not cover training the actual machine learning model and exporting it to ONNX. Please see section [YOLOv8](https://github.com/FishCenSV2/fishCenSV2/tree/main?tab=readme-ov-file#yolov8) for this**

## Table of Contents

## Function Breakdowns

**NOTE: The below assumes that the engine file already exists. To see how to create the engine file from the ONNX file please see the section [TensorRT](https://github.com/FishCenSV2/fishCenSV2/tree/main?tab=readme-ov-file#tensorrt).**

### Initialization
---
The following shows the `init` function.

```cpp
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
```

Much of the setup here should be left well alone. However, the following lines are very important

```cpp
_input_index = _engine->getBindingIndex("images");
_output_index = _engine->getBindingIndex("output0");
```

**IMPORTANT!** These strings might change depending on the model. In order to find the correct names you can drag-and-drop the ONNX model into [Netron](https://netron.app/). The following figure shows what you should see at the very top.

IMAGE GOES HERE

Click on the highlighted box and you should see a panel pop up on the right. Look for the two bolded titles `INPUTS` and `OUTPUTS`. These are what you substitute for the strings in the argument of the `getBindingIndex` function. The following figure shows how I obtained `"images"` and `"output0"`.

IMAGE GOES HERE. 

Next we can get the input and output dimensions `idims` and `odims`

```cpp
const auto idims = _engine->getBindingDimensions(_input_index);
const auto odims = _engine->getBindingDimensions(_output_index);
```

**IMPORTANT!** Your model might have dynamic input sizing (which also means dynamic output sizing) which means your model can take any input size. This can be seen by the sign of `idims.d[0]` and `odims.d[0]` being negative. This is why we generally start from `idims.d[1]` or `odims.d[1]`.

The next part of the code finds the max output dimension and the total dimension length from `odims`.

```cpp
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
```

For example, using the standard COCO classes, the `yolov8n.pt` weights, and a input size of 320x320x3 images, the output size is 84x2100 (you can also confirm this with Netron). So the max dimension is 2100 and the total dimension length is 84 multiplied by 2100 which is 176400. 

Finally, we allocate buffers on the GPU as well as create the output array whose size is the total dimension length from before.

```cpp
cudaMalloc(&_buffers[_input_index], input_len * sizeof(float));
cudaMalloc(&_buffers[_output_index], output_len * sizeof(float));

_output_data = std::make_unique<float[]>(output_len);
```

The `_buffers` member variable is a `void*` buffer of size two. It stores the input and output data for the GPU.

### Preprocessing 
---
The following shows the `preprocess` function.

```cpp
cv::Mat YoloV8::preprocess(cv::Mat& image) {
    cv::Mat frame;

    //The padding is hardcoded based on an input of 640*480 so it will just
    //add 80 to the top and bottom of the the rows to make it 640*640
    cv::copyMakeBorder(image,frame, 80,80,0,0, CV_HAL_BORDER_CONSTANT, cv::Scalar(0,0,0));      

    /*
    TensorRT prefers NCHW but apparently OpenCV uses NHCW. We can use
    this function to also do some other pre-processing like scaling
    the input to be between 0 and 1 instead of 0 and 255.

    NOTE: The cv::Size(320,320) is also hardcoded based on the model.
          This should realistically be determined using idims.d 
    */
    return cv::dnn::blobFromImage(frame,1.0/255,cv::Size(320,320),cv::Scalar(0,0,0),true,false,CV_32F);

}
```

The comments should make most of this clear. The `cv::dnn::blobFromImage` functions parameters are shown below. The output of this function should be directly passed to the `predict` function.

```cpp
Mat cv::dnn::blobFromImages	(InputArrayOfArrays 	images,
                            double 	                scalefactor = 1.0,
                            Size 	                size = Size(),
                            const Scalar& 	        mean = Scalar(),
                            bool 	                swapRB = false,
                            bool 	                crop = false,
                            int 	                ddepth = CV_32F 
)	
```
### Preprocessing 
---
The following shows the `predict` function.

```cpp
void YoloV8::predict(cv::Mat& input) {
    cudaMemcpy(_buffers[_input_index], input.data, input.total() * input.elemSize(),cudaMemcpyHostToDevice);

    _context->executeV2(_buffers);

    cudaMemcpy(_output_data.get(), _buffers[_output_index], 
        output_len * sizeof(float),
        cudaMemcpyDeviceToHost);
        
}
```

What this essentially does is 
- Copy the preprocessed input image to the GPU using the `_buffers` array.
- Run the engine on the input.
- Copy the output from the GPU to the `_output_data` array.

The output needs to be processed so the `postprocess` function **must** be called after this.

### Postprocessing
The following shows the `postprocess` function.

```cpp
std::vector<Object> YoloV8::postprocess(float scale_factor) {

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

    return objects;

}
```

The `_output_data` array (1D) has dimensions of `(# of classes + 4) x (# of objects detectable)`. The 4 refers to the xywh which specifies the bounding box dimensions. For example if the output dimensions are 84x2100 then we have 80 classes and 2100 detectable objects. In order to easily index through this array we use a `cv::Mat` called `data` which is not a copy of `_output_data`, but rather has a reference to it and allows for syntax similar to accessing a 2D array.

To visualize the output in 2D we will assume our output has dimensions of 5x2 for simplicity.


|          | Object 1 | Object 2 |
|----------|----------|----------|
| x        |   210    |   110    |
| y        |   320    |   321    |
| w        |   100    |    50    |
| h        |    60    |   120    |
| Class 1  |     4    |    12    |
| Class 2  |    35    |    0     |

For each object or column we have the 4 values specifying the bounding box location and size as well as 2 numbers which correspond to unnormalized scores for each class. We decide the class of the object by picking the one with the highest number. The following code does the equivalent of this

```cpp
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
    }
}
```

Once we have found the max score for a object we compare it to a threshold value and if it is greater than this threshold we add it to three vectors. The final step is to peform non-maximum suppression (NMS) to remove duplicate detections which returns a vector of indices that we store in `_indices`. The indices correspond to indices in the three vectors of objects that we should keep. For example, a vector containing indices 0,3,4 means we should use elements from _boxes[0], _boxes[3], _boxes[4], _class_ids[0], _class_ids[3], etc. 

Finally, we can use these elements from the three vectors to create a vector of bounding box objects. The objects are structs which contain the elements from the three vectors as shown below.

```cpp
//Loop over all indices to only get the non-duplicates.
for (int i = 0; i < _indices.size(); i++) {
    int index = _indices[i];
    objects.emplace_back();
    objects[i].rect = _boxes[index];
    objects[i].label = _class_ids[index];
    objects[i].prob = _confidences[index];
}
```

This vector is then returned to the user.

