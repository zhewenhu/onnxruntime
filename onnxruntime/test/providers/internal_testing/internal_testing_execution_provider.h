// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <set>
#include "core/framework/execution_provider.h"

namespace onnxruntime {
class NodeArg;

class InternalTestingExecutionProvider : public IExecutionProvider {
 public:
  InternalTestingExecutionProvider(const std::unordered_set<std::string>& ops,
                                   const std::unordered_set<std::string>& stop_ops = {},
                                   int get_capability_version = 0,
                                   bool debug_output = false);
  virtual ~InternalTestingExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability3(const onnxruntime::GraphViewer& graph_view,
                 const std::vector<const KernelRegistry*>& /*kernel_registries*/) const;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  FusionStyle GetFusionStyle() const override {
    return FusionStyle::FilteredGraphViewer;
  }

 private:
  std::unique_ptr<ComputeCapability> MakeComputeCapability(const GraphViewer& graph_viewer,
                                                           const std::unordered_set<const NodeArg*>& graph_outputs,
                                                           const std::vector<const Node*>& group) const;
  const std::unordered_set<std::string> ops_;
  const int get_capability_version_;
  const bool debug_output_;

  // operators that we stop processing at. e.g. NonMaxSuppression is the beginning of post-processing and for NNAPI
  // we don't want to go back to NNAPI for that phase.
  const std::unordered_set<std::string> stop_ops_;
};
}  // namespace onnxruntime
