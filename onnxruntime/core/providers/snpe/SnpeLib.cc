#include "SnpeLib.h"
#include "core/common/common.h"

#include <iostream>
#include <unordered_map>

  std::unique_ptr<zdl::SNPE::SNPE> SnpeLib::InitializeSnpe(zdl::DlContainer::IDlContainer* container,
                                                           const std::vector<std::string>* output_tensor_names,
                                                           const std::vector<std::string>* input_tensor_names) {
    zdl::SNPE::SNPEBuilder snpe_builder(container);

    zdl::DlSystem::StringList dl_output_tensor_names = {};
    if ((nullptr != output_tensor_names) && (output_tensor_names->size() != 0)) {
      for (auto layerName : *output_tensor_names) {
        dl_output_tensor_names.append(layerName.c_str());
      }
    }

    std::unique_ptr<zdl::SNPE::SNPE> snpe = snpe_builder.setOutputTensors(dl_output_tensor_names).setRuntimeProcessor(runtime_).build();

    input_tensor_map_.clear();
    input_tensors_.clear();
    if ((snpe != nullptr) && (input_tensor_names != nullptr) && (input_tensor_names->size() != 0)) {
      input_tensors_.resize(input_tensor_names->size());
      for (size_t i = 0; i < input_tensor_names->size(); ++i) {
        zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> input_shape = snpe->getInputDimensions(input_tensor_names->at(i).c_str());
        if (!input_shape) {
          LOGS_DEFAULT(ERROR) << "Snpe cannot get input shape for input name: " << input_tensor_names->at(i).c_str();
          input_tensor_map_.clear();
          input_tensors_.clear();
          return nullptr;
        }
        input_tensors_[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*input_shape);
        zdl::DlSystem::ITensor* input_tensor = input_tensors_[i].get();
        if (!input_tensor) {
          LOGS_DEFAULT(ERROR) << "Snpe cannot create ITensor";
          input_tensor_map_.clear();
          input_tensors_.clear();
          return nullptr;
        }
        input_tensor_map_.add(input_tensor_names->at(i).c_str(), input_tensor);
      }
    }

    return snpe;
  }

  bool SnpeLib::Initialize(const char* dlcPath, const std::vector<std::string>* output_layer_names,
                  const std::vector<std::string>* input_layer_names) {
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlcPath));
    if (!container) {
      LOGS_DEFAULT(ERROR) << "failed open " << dlcPath << " container file";
      return false;
    }

    snpe_ = InitializeSnpe(container.get(), output_layer_names, input_layer_names);
    if (!snpe_)
    {
      LOGS_DEFAULT(ERROR) << "failed to build snpe";
      return false;
    }

    return true;
  }

  bool SnpeLib::Initialize(const unsigned char* dlcData, size_t size,
                           const std::vector<std::string>* output_layer_names,
                           const std::vector<std::string>* input_layer_names) {
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(dlcData, size);
    if (container == nullptr)
    {
      LOGS_DEFAULT(ERROR) << "failed open container buffer";
      return false;
    }

    snpe_ = InitializeSnpe(container.get(), output_layer_names, input_layer_names);
    if (!snpe_)
    {
      LOGS_DEFAULT(ERROR) << "failed to build snpe " << zdl::DlSystem::getLastErrorString();
      return false;
    }

    return true;
  }

  bool SnpeLib::SnpeProcessMultipleOutput(const unsigned char* input, size_t input_size, size_t output_number, unsigned char* outputs[], size_t output_sizes[]) {
    try {
      zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> input_shape = snpe_->getInputDimensions();
      if (!input_shape) {
        LOGS_DEFAULT(ERROR) << "Snpe cannot get input shape";
        return false;
      }
      std::unique_ptr<zdl::DlSystem::ITensor> input_tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*input_shape);
      if (!input_tensor) {
        LOGS_DEFAULT(ERROR) << "Snpe cannot create ITensor";
        return false;
      }
      // ensure size of the input buffer matches input shape buffer size
      size_t input_data_size = input_tensor->getSize() * sizeof(float);
      if (input_data_size != input_size) {
        LOGS_DEFAULT(ERROR) << "Snpe input size incorrect: expected " << input_data_size << "given " << input_size << " bytes";
        return false;
      }
      memcpy(input_tensor->begin().dataPointer(), input, input_size);

      zdl::DlSystem::TensorMap output_tensor_map;
      bool result = snpe_->execute(input_tensor.get(), output_tensor_map);
      if (!result) {
        LOGS_DEFAULT(ERROR) << "Snpe Error while executing the network.";
        return false;
      }
      if (output_tensor_map.size() == 0) {
        return false;
      }

      zdl::DlSystem::StringList tensor_names = output_tensor_map.getTensorNames();

      for (size_t i=0; i < output_number; i++) {
        zdl::DlSystem::ITensor* tensor = output_tensor_map.getTensor(tensor_names.at(i));
        // ensure size of the output buffer matches output shape buffer size
        size_t output_data_size = tensor->getSize() * sizeof(float);
        if (output_data_size > output_sizes[i]) {
          LOGS_DEFAULT(ERROR) << "Snpe output size incorrect: output_layer: " << tensor_names.at(i) << " expected "
                              << output_data_size << " given " << output_sizes[i] << " bytes.";
          return false;
        }
        memcpy(outputs[i], tensor->cbegin().dataPointer(), output_data_size);
      }

      return true;
    }
    catch (...){
      LOGS_DEFAULT(ERROR) << "Snpe threw exception";
      return false;
    }
  }


  bool SnpeLib::SnpeProcess(const unsigned char* input, size_t input_size, unsigned char* output, size_t output_size) {
    // Use SnpeProcessMultipleOutput with 1 output layer
    const int output_layer = 1;
    unsigned char* outputs_array[output_layer];
    size_t output_sizes_array[output_layer];
    outputs_array[0] = output;
    output_sizes_array[0] = output_size;
    return SnpeProcessMultipleOutput(input, input_size, output_layer, outputs_array, output_sizes_array);
  }


  bool SnpeLib::SnpeProcessMultipleInputsMultipleOutputs(const unsigned char** inputs, const size_t* input_sizes, size_t input_number,
                                                         unsigned char** outputs, const size_t* output_sizes, size_t output_number) {
    try {
      if (input_number != input_tensors_.size()) {
        LOGS_DEFAULT(ERROR) << "Snpe number of inputs doesn't match";
        return false;
      }
      for (size_t i=0; i < input_number; ++i) {
        zdl::DlSystem::ITensor* input_tensor = input_tensors_[i].get();
        // ensure size of the input buffer matches input shape buffer size
        size_t input_data_size = input_tensor->getSize() * sizeof(float);
        if (input_data_size != input_sizes[i]) {
          LOGS_DEFAULT(ERROR) << "Snpe input size incorrect: expected %d, given %d bytes" << input_data_size << input_sizes[i];
          return false;
        }
        memcpy(input_tensor->begin().dataPointer(), inputs[i], input_sizes[i]);
      }
      zdl::DlSystem::TensorMap output_tensor_map;
      bool result = snpe_->execute(input_tensor_map_, output_tensor_map);
      if (!result) {
        LOGS_DEFAULT(ERROR) << "Snpe Error while executing the network.";
        return false;
      }
      if (output_tensor_map.size() == 0) {
        return false;
      }

      zdl::DlSystem::StringList tensor_names = output_tensor_map.getTensorNames();

      for (size_t i=0; i < output_number; i++)
      {
        zdl::DlSystem::ITensor* tensor = output_tensor_map.getTensor(tensor_names.at(i));
        // ensure size of the output buffer matches output shape buffer size
        size_t output_data_size = tensor->getSize() * sizeof(float);
        if (output_data_size > output_sizes[i]) {
          LOGS_DEFAULT(ERROR) << "Snpe output size incorrect: output_layer" << tensor_names.at(i) << " expected "
                              << output_data_size << " given " << output_sizes[i] << " bytes";
          return false;
        }
        memcpy(outputs[i], tensor->cbegin().dataPointer(), output_data_size);
      }

      return true;
    }
    catch (...){
      LOGS_DEFAULT(ERROR) << "Snpe threw exception";
      return false;
    }
  }
