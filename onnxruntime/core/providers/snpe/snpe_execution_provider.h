// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/providers/snpe/snpe_provider_factory.h"
#include "SnpeLib.h"

namespace onnxruntime {

class SNPEExecutionProvider : public IExecutionProvider {
 public:
  SNPEExecutionProvider(bool enforce_dsp);
  virtual ~SNPEExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& ) const override;
    
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  bool GetEnforceDsp() const { return enforce_dsp_; }

 private:
  bool enforce_dsp_;
  std::unique_ptr<SnpeLib> snpe_rt_;
};
}  // namespace onnxruntime
