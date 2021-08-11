#include <iostream>
#include "onnxruntime_cxx_api.h"

struct Input {
    const char* name = nullptr;
    std::vector<int64_t> dims;
    std::vector<float> values;
};

struct OrtTensorDimensions : std::vector<int64_t> {
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
        OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
};

struct MultithresholdKernel {
    MultithresholdKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info) : ort_(ort) {
        out_dtype_ = ort_.KernelInfoGetAttribute<std::string>(info, "out_dtype");
        out_scale_ = ort_.KernelInfoGetAttribute<float>(info, "out_scale");
        out_bias_ = ort_.KernelInfoGetAttribute<float>(info, "out_bias");
        data_layout_ = ort_.KernelInfoGetAttribute<std::string>(info, "data_layout");
    }

    void Compute(OrtKernelContext* context);

private:
    Ort::CustomOpApi ort_;
    std::string out_dtype_;
    float out_scale_;
    float out_bias_;
    std::string data_layout_;
};

struct MultithresholdOp : Ort::CustomOpBase<MultithresholdOp, MultithresholdKernel> {

    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new MultithresholdKernel(api, info); };
    const char* GetName() const { return "MultiThreshold"; };

    size_t GetInputTypeCount() const { return 2; };
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
        // The second input (index == 1) is optional
//        if (index == 1)
//            return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    }

    size_t GetOutputTypeCount() const { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
    OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    }
};

#include "multithreshold.cc"
