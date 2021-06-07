// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "internal_testing_execution_provider.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/graph/model.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

#include <queue>

namespace onnxruntime {

constexpr const char* INTERNAL_TESTING_EP = "InternalTestingEP";

InternalTestingExecutionProvider::InternalTestingExecutionProvider(const std::unordered_set<std::string>& ops,
                                                                   const std::unordered_set<std::string>& stop_ops,
                                                                   int get_capability_version,
                                                                   bool debug_output)
    : IExecutionProvider{utils::kInternalTestingExecutionProvider, true},
      ep_name_{INTERNAL_TESTING_EP},
      ops_{ops},
      stop_ops_{stop_ops},
      get_capability_version_{get_capability_version},
      debug_output_{debug_output} {
  //
  // TODO: Allocation planner calls GetAllocator for the individual EP. It would be better if it goes through
  // the session state to get the allocator so it's per-device (or for the allocation planner to try the EP first
  // and fall back to using session state next by passing in a functor it can use to call SessionState::GetAllocator).

  AllocatorCreationInfo device_info(
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(INTERNAL_TESTING_EP,
                                                            OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));
}

InternalTestingExecutionProvider::~InternalTestingExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
InternalTestingExecutionProvider::GetCapability3(const onnxruntime::GraphViewer& graph_viewer,
                                                 const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // find all supported nodes first
  std::unordered_set<const Node*> supported_nodes;

  const auto& topo_nodes = graph_viewer.GetNodesInTopologicalOrder();
  std::for_each(topo_nodes.cbegin(), topo_nodes.cend(),
                [this, &supported_nodes, &graph_viewer](NodeIndex node_index) {
                  const Node* node = graph_viewer.GetNode(node_index);
                  bool supported = ops_.count(node->OpType()) != 0;
                  if (supported) {
                    supported_nodes.insert(node);
                  }
                });

  if (supported_nodes.empty()) {
    return {};
  }

  // NOTE: GetCapability is called for all subgraphs from the bottom up, for one execution provider at a time.
  //       i.e. each execution provider will see the entire model individually.
  // If your execution provider may selectively handle a control flow node (Scan/Loop/If) if it can process all nodes
  // in the subgraph, here would be the place to check if:
  //   - you're processing a subgraph (graph_viewer.IsSubgraph() returns true)
  //   - and all nodes are handled (supported_nodes.size == graph_viewer.NumberOfNodes())
  //
  // If that is the case and you wish to take the control flow node containing the subgraph:
  //   - return an empty vector so the nodes are left as is
  //   - note the node containing the subgraph (graph_viewer.ParentNode()) so that when GetCapability is called for
  //     the parent node you can either:
  //     - include that node in supported_nodes if your Compile implementation can handle it potentially
  //       being part of a larger partition; or
  //     - create a ComputeCapability instance for just the control flow node by calling utils::MakeComputeCapability
  //       and adding the instance to the partitions returned by CreateSupportedPartitions

  auto generate_metadef_name = [this, &graph_viewer]() {
    uint64_t model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
    return ep_name_ + "_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
  };

  return utils::CreateSupportedPartitions(graph_viewer, supported_nodes, stop_ops_,
                                          generate_metadef_name, ep_name_, debug_output_);
}

std::vector<std::unique_ptr<ComputeCapability>>
InternalTestingExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                const std::vector<const KernelRegistry*>& /*registries*/) const {
  if (get_capability_version_ == 3) {
    return GetCapability3(graph_viewer, {});
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;

  /* 
  Very basic search for groups of nodes that can be handled by the EP.
  This doesn't work perfectly if you have a scenario like the following where A and D could be handled by the EP
  but B is between them in the topological sort as you'll get two single node capabilities. However if can also
  be advantageous if C and E could be handled by the EP as they would be combined with D even though not connected.
  Not sure how often each of these scenarios happens. 

    A  B  C
    | /   |
    D     E
    |     |
    
  Would probably be better to walk the edges for each node the EP can handle as they are iterated in topological order,
  accumulating nodes (and saving which ones have been taken) until you run out. This would guarantee all
  connected nodes that can be handled are grouped together.     
  */

  std::vector<std::vector<const Node*>> node_groups;
  std::vector<const Node*> cur_group;

  for (NodeIndex node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const Node* node = graph_viewer.GetNode(node_index);  // never nullptr
    if (ops_.count(node->OpType())) {
      cur_group.push_back(node);
    } else if (!cur_group.empty()) {
      node_groups.push_back(std::move(cur_group));
    }
  }

  if (!cur_group.empty()) {
    node_groups.push_back(std::move(cur_group));
  }

  if (node_groups.empty()) {
    return result;
  }

  for (const auto& group : node_groups) {
    auto generate_metadef_name = [this, &graph_viewer]() {
      uint64_t model_hash;
      int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
      auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
      return ep_name_ + "_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
    };

    result.push_back(utils::MakeComputeCapability(graph_viewer, group, generate_metadef_name, ep_name_));
  }

  return result;
}

common::Status InternalTestingExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                                                         std::vector<NodeComputeInfo>& node_compute_funcs) {
  // Create a function to generate dummy empty output for each fused node so the model can be executed.
  for (const auto& node_and_viewer : fused_nodes) {
    NodeComputeInfo compute_info;
    const Node& node = node_and_viewer.fused_node;

    //{
    //  const GraphViewer& graph_viewer = node_and_viewer.filtered_graph;
    //  DEBUG_LOGS(debug_output_)<< "Fusing nodes: ";
    //  for (const auto& unfused_node : graph_viewer.Nodes()) {
    //    DEBUG_LOGS(debug_output_)<< " '" << unfused_node.Name() << "':" << unfused_node.Index();
    //  }
    //  DEBUG_LOGS(debug_output_)<< std::endl;
    //}

    compute_info.create_state_func = [](ComputeContext* /*context*/, FunctionState* /*state*/) {
      return 0;
    };

    compute_info.release_state_func = [](FunctionState /*state*/) {
    };

    compute_info.compute_func = [&node](FunctionState /*state*/, const OrtCustomOpApi* c_api,
                                        OrtKernelContext* context) -> Status {
      Ort::CustomOpApi api{*c_api};  // use C++ API for convenience

      const auto outputs = node.OutputDefs();
      const size_t num_outputs = outputs.size();

      for (size_t i = 0; i < num_outputs; i++) {
        const auto* shape_proto = outputs[i]->Shape();
        if (shape_proto == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown output shapes are not supported");
        }

        TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*shape_proto);
        if (shape.Size() < 0) {
          // arbitrarily set any unknown dim to 1
          for (size_t idx = 0, end = shape.NumDimensions(); idx < end; ++idx) {
            if (shape[idx] == -1) {
              shape[idx] = 1;
            }
          }
        }

        // create the output_tensor.
        auto* ortvalue = api.KernelContext_GetOutput(context, i, shape.GetDims().data(), shape.GetDims().size());

        // and fill with zeros
        auto* tensor = ortvalue->GetMutable<Tensor>();
        void* data = tensor->MutableDataRaw();
        auto bytes = tensor->SizeInBytes();
        memset(data, 0, bytes);
      };

      return Status::OK();
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}
}  // namespace onnxruntime
