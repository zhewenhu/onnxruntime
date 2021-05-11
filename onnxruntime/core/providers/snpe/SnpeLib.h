#pragma once

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlError.hpp"

#include <memory>
#include <string>
#include <vector>
#include "core/common/logging/macros.h"
#include "core/common/logging/logging.h"

static std::string s_getRuntimeString(const zdl::DlSystem::Runtime_t& t) {
  std::unordered_map<zdl::DlSystem::Runtime_t, std::string> s_names;
  s_names[zdl::DlSystem::Runtime_t::AIP_FIXED8_TF] = "AIP_FIXED8_TF";
  s_names[zdl::DlSystem::Runtime_t::DSP_FIXED8_TF] = "DSP_FIXED8_TF";
  s_names[zdl::DlSystem::Runtime_t::GPU_FLOAT16] = "GPU_FLOAT16";
  s_names[zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID] = "GPU_FLOAT32_16_HYBRID";
  s_names[zdl::DlSystem::Runtime_t::CPU_FLOAT32] = "CPU_FLOAT32";
  if (s_names.find(t) != s_names.end()) {
    return s_names[t];
  }
  return "RUNTIME_UNKNOWN";
}

static zdl::DlSystem::Runtime_t s_getPreferredRuntime(bool enforce_dsp, bool device_uses_dsp_only, bool device_must_not_use_dsp) {
  zdl::DlSystem::Runtime_t runtimes[] = {zdl::DlSystem::Runtime_t::DSP_FIXED8_TF,
                                         zdl::DlSystem::Runtime_t::AIP_FIXED8_TF,
                                         zdl::DlSystem::Runtime_t::GPU_FLOAT16,
                                         zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID,
                                         zdl::DlSystem::Runtime_t::CPU_FLOAT32};
  static zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
  zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
  LOGS_DEFAULT(INFO) << "SNPE Version %s" << version.asString().c_str();

  bool ignore_dsp = device_must_not_use_dsp | !enforce_dsp;
  bool ignore_others = device_uses_dsp_only & enforce_dsp;
  int start = ignore_dsp * 2;
  int end = ignore_others ? 2 : sizeof(runtimes) / sizeof(*runtimes);

  if (ignore_others) {
    runtime = zdl::DlSystem::Runtime_t::DSP;
  }
  // start with skipping aip and dsp if specified.
  for (int i = start; i < end; ++i) {
    LOGS_DEFAULT(INFO) << "testing runtime %d" << (int)runtimes[i];
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtimes[i])) {
      runtime = runtimes[i];
      break;
    }
  }
  LOGS_DEFAULT(INFO) << "using runtime %d" << (int)runtime;
  return runtime;
}

class SnpeLib {

 public:
  SnpeLib(bool enforce_dsp, bool device_uses_dsp_only, bool device_must_not_use_dsp)
      : runtime_(zdl::DlSystem::Runtime_t::CPU) {
#if defined(_WIN32)
    (enforce_dsp);
    (device_uses_dsp_only);
    (device_must_not_use_dsp);
#if !defined(_M_ARM64)
    runtime_ = zdl::DlSystem::Runtime_t::CPU;
#else
    if (enforce_dsp) {
      // force DSP on ARM64 WIN32
      runtime_ = zdl::DlSystem::Runtime_t::DSP;
    }
#endif
#else
    // ANDROID
    runtime_ = s_getPreferredRuntime(enforce_dsp, device_uses_dsp_only, device_must_not_use_dsp);
#endif
    LOGS_DEFAULT(INFO) << "SNPE using runtime %s" << s_getRuntimeString(runtime_);
  }
  ~SnpeLib(){};
  
  bool SnpeProcess(const unsigned char* input, size_t input_size, unsigned char* output, size_t output_size);
  bool SnpeProcessMultipleOutput(const unsigned char* input, size_t input_size, size_t output_number, unsigned char* outputs[], size_t output_sizes[]);
  bool SnpeProcessMultipleInputsMultipleOutputs(const unsigned char** inputs, const size_t* input_sizes, size_t input_number,
                                                        unsigned char** outputs, const size_t* output_sizes, size_t output_number);

  std::unique_ptr<zdl::SNPE::SNPE> InitializeSnpe(zdl::DlContainer::IDlContainer* container,
                                                  const std::vector<std::string>* output_tensor_names = nullptr,
                                                  const std::vector<std::string>* input_tensor_names = nullptr);
   bool Initialize(const char* dlcPath, const std::vector<std::string>* output_layer_names = nullptr,
                   const std::vector<std::string>* input_layer_names = nullptr);
   bool Initialize(const unsigned char* dlcData, size_t size, const std::vector<std::string>* output_layer_names = nullptr,
                  const std::vector<std::string>* input_layer_names = nullptr);

  private:
   zdl::DlSystem::Runtime_t runtime_;
   std::unique_ptr<zdl::SNPE::SNPE> snpe_;
   std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> input_tensors_;
   zdl::DlSystem::TensorMap input_tensor_map_;
};

static std::unique_ptr<SnpeLib> SnpeLibFactory(const unsigned char* dlc_data, size_t size, const std::vector<std::string>* output_layer_names = nullptr,
                                               bool enforce_dsp = true, const std::vector<std::string>* input_layer_names = nullptr,
                                               bool device_uses_dsp_only = false, bool device_must_not_use_dsp = false) {
  std::unique_ptr<SnpeLib> object(new SnpeLib(enforce_dsp, device_uses_dsp_only, device_must_not_use_dsp));

  if (!object) {
    ORT_THROW("failed to make snpe library");
  }

  if (!object->Initialize(dlc_data, size, output_layer_names, input_layer_names)) {
    ORT_THROW("failed to initialize dlc from buffer");
  }

  return object;
}
