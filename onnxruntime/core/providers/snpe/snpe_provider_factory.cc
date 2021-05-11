// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/snpe/snpe_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "snpe_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {
struct SNPEProviderFactory : IExecutionProviderFactory {
  SNPEProviderFactory(bool enforce_dsp)
      : enforce_dsp_(enforce_dsp) {}
  ~SNPEProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  bool enforce_dsp_;
};

std::unique_ptr<IExecutionProvider> SNPEProviderFactory::CreateProvider() {
  return std::make_unique<SNPEExecutionProvider>(enforce_dsp_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_SNPE(bool enforce_dsp) {
  return std::make_shared<onnxruntime::SNPEProviderFactory>(enforce_dsp);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_SNPE,
                    _In_ OrtSessionOptions* options, bool enforce_dsp) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_SNPE(enforce_dsp));
  return nullptr;
}
