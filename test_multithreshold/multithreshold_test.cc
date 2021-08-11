#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <cassert>
#include "multithreshold.h"
#include "onnxruntime_cxx_api.h"

typedef const char *PATH_TYPE;
#define TSTR(X) (X)
static constexpr PATH_TYPE MODEL_URI = TSTR("../testdata/test.onnx");

int main(int argc, char **argv) {

    Ort::Env ort_env = Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "Default");

    MultithresholdOp custom_op;

    Ort::CustomOpDomain custom_op_domain("finn.custom_op.general");
    custom_op_domain.Add(&custom_op);

    Ort::SessionOptions session_options;
    session_options.Add(custom_op_domain);

    std::vector<Input> inputs(2);
    // First input v
    auto input = inputs.begin();
    input->name = "v";
    input->dims = {6, 3, 2, 2};
    input->values = {
            4.8,
            3.2,
            1.2,
            4.9,
            7.8,
            2.4,
            3.1,
            4.7,
            6.2,
            5.1,
            4.9,
            2.2,
            6.2,
            0.0,
            0.8,
            4.7,
            0.2,
            5.6,
            8.9,
            9.2,
            9.1,
            4.0,
            3.3,
            4.9,
            2.3,
            1.7,
            1.3,
            2.2,
            4.6,
            3.4,
            3.7,
            9.8,
            4.7,
            4.9,
            2.8,
            2.7,
            8.3,
            6.7,
            4.2,
            7.1,
            2.8,
            3.1,
            0.8,
            0.6,
            4.4,
            2.7,
            6.3,
            6.1,
            1.4,
            5.3,
            2.3,
            1.9,
            4.7,
            8.1,
            9.3,
            3.7,
            2.7,
            5.1,
            4.2,
            1.8,
            4.1,
            7.3,
            7.1,
            0.4,
            0.2,
            1.3,
            4.3,
            8.9,
            1.4,
            1.6,
            8.3,
            9.4
    };

    // Second input thresholds
    input = std::next(input, 1);
    input->name = "thresholds";
    input->dims = {3, 7};
    input->values = {
            0.8,
            1.4,
            1.7,
            3.5,
            5.2,
            6.8,
            8.2,
            0.2,
            2.2,
            3.5,
            4.5,
            6.6,
            8.6,
            9.2,
            1.3,
            4.1,
            4.5,
            6.5,
            7.8,
            8.1,
            8.9,
    };

    std::vector<Ort::Value> ort_inputs;
    std::vector<const char *> input_names;
    Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

    for (size_t i = 0; i < inputs.size(); i++) {
        input_names.emplace_back(inputs[i].name);
        ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(info, const_cast<float *>(inputs[i].values.data()),
                                                                inputs[i].values.size(), inputs[i].dims.data(),
                                                                inputs[i].dims.size()));
    }

    const char *output_name = "results";

//    ort_inputs.erase(ort_inputs.begin() + 2);  // remove the last input in the container
    Ort::Session session(ort_env, MODEL_URI, session_options);
    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                   &output_name, 1);
        assert(ort_outputs.size() == 1u);

    // Validate results
    std::vector<int64_t> y_dims = {6, 3, 2, 2};
    std::vector<float> values_y = {
            4.0,
            3.0,
            1.0,
            4.0,
            5.0,
            2.0,
            2.0,
            4.0,
            3.0,
            3.0,
            3.0,
            1.0,
            5.0,
            0.0,
            1.0,
            4.0,
            1.0,
            4.0,
            6.0,
            7.0,
            7.0,
            1.0,
            1.0,
            3.0,
            3.0,
            3.0,
            1.0,
            3.0,
            4.0,
            2.0,
            3.0,
            7.0,
            3.0,
            3.0,
            1.0,
            1.0,
            7.0,
            5.0,
            4.0,
            6.0,
            2.0,
            2.0,
            1.0,
            1.0,
            2.0,
            1.0,
            3.0,
            3.0,
            2.0,
            5.0,
            3.0,
            3.0,
            4.0,
            5.0,
            7.0,
            3.0,
            1.0,
            3.0,
            2.0,
            1.0,
            4.0,
            6.0,
            6.0,
            0.0,
            1.0,
            1.0,
            3.0,
            6.0,
            1.0,
            1.0,
            6.0,
            7.0
    };
    auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
        assert(type_info.GetShape() == y_dims);
    size_t total_len = type_info.GetElementCount();
        assert(values_y.size() == total_len);

    auto *f = ort_outputs[0].GetTensorMutableData<float>();
    for (size_t i = 0; i != total_len; ++i) {
        assert(values_y[i] == f[i]);
//        printf("%f\n", f[i]);
    }

    return 0;
}