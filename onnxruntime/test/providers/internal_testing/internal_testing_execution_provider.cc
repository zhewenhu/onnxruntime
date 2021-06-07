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
#include "core/session/onnxruntime_cxx_api.h"

#include <queue>

namespace onnxruntime {

constexpr const char* INTERNAL_TESTING_EP = "InternalTestingEP";

InternalTestingExecutionProvider::InternalTestingExecutionProvider(const std::unordered_set<std::string>& ops,
                                                                   const std::unordered_set<std::string>& stop_ops,
                                                                   int get_capability_version,
                                                                   bool debug_output)
    : IExecutionProvider{utils::kInternalTestingExecutionProvider, true},
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

// in a debug build, log at VERBOSE level if debug output is enabled
// in a release build, throw away the log statements completely (so they don't impact on binary size)
#ifndef NDEBUG
#define DEBUG_LOGS(debug_output) \
  debug_output&& std::cout
#else
#define DEBUG_LOGS(debug_output) \
  false && std::cout
#endif

static std::vector<NodeIndex> PartitionAwareTopoSort(const GraphViewer& graph_viewer,
                                                     const std::unordered_set<const Node*>& supported_nodes) {
  std::queue<const Node*> supported_to_visit, unsupported_to_visit;
  std::unordered_map<NodeIndex, size_t> in_degree;
  std::vector<NodeIndex> topo_order;

  auto num_nodes = graph_viewer.NumberOfNodes();
  topo_order.reserve(num_nodes);
  in_degree.reserve(num_nodes);

  auto add_to_visit = [&](const Node& node) {
    if (supported_nodes.count(&node)) {
      supported_to_visit.push(&node);
    } else {
      unsupported_to_visit.push(&node);
    }
  };

  // root nodes
  for (auto& node : graph_viewer.Nodes()) {
    size_t input_edge_count = node.GetInputEdgesCount();
    in_degree.insert({node.Index(), input_edge_count});
    if (input_edge_count == 0) {
      add_to_visit(node);
    }
  }

  // prefer unsupported nodes first. this will increase the number of inputs potentially available to the first
  // partition handled by this EP.
  bool processing_supported_nodes = false;

  while (!supported_to_visit.empty() || !unsupported_to_visit.empty()) {
    const Node* current = nullptr;

    // see if we need to flip
    if ((processing_supported_nodes && supported_to_visit.empty()) ||
        (!processing_supported_nodes && unsupported_to_visit.empty())) {
      processing_supported_nodes = !processing_supported_nodes;
      continue;
    }

    // get next node from same partition
    if (processing_supported_nodes) {
      current = supported_to_visit.front();
      supported_to_visit.pop();
    } else {
      current = unsupported_to_visit.front();
      unsupported_to_visit.pop();
    }

    for (auto node_it = current->OutputNodesBegin(), end = current->OutputNodesEnd(); node_it != end; ++node_it) {
      in_degree[node_it->Index()]--;

      if (in_degree[node_it->Index()] == 0) {
        add_to_visit(*node_it);
      }
    }

    topo_order.push_back(current->Index());
  }

  if (graph_viewer.NumberOfNodes() != static_cast<int>(topo_order.size())) {
    ORT_THROW("LOGIC ERROR. Graph was valid when loaded but the topological sort has produced invalid output.");
  }

  return topo_order;
}

static std::unordered_set<const Node*> CreateExcludedNodeSet(const GraphViewer& graph_viewer,
                                                             const std::unordered_set<std::string>& stop_ops) {
  std::unordered_set<const Node*> excluded_nodes;
  const auto end_stop_ops = stop_ops.cend();

  for (const NodeIndex node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const Node& node = *graph_viewer.GetNode(node_index);

    if (excluded_nodes.find(&node) == excluded_nodes.cend() &&
        stop_ops.find(node.OpType()) != end_stop_ops) {
      excluded_nodes.insert(&node);

      // add all the downstream nodes
      std::queue<const Node*> nodes_to_process;
      nodes_to_process.push(&node);

      while (!nodes_to_process.empty()) {
        const Node* cur_node = nodes_to_process.front();
        nodes_to_process.pop();

        std::for_each(cur_node->OutputNodesBegin(), cur_node->OutputNodesEnd(),
                      [&nodes_to_process, &excluded_nodes](const Node& output_node) {
                        nodes_to_process.push(&output_node);
                        excluded_nodes.insert(&output_node);
                      });
      }
    }
  }

  return excluded_nodes;
}

std::unique_ptr<ComputeCapability>
InternalTestingExecutionProvider::MakeComputeCapability(const GraphViewer& graph_viewer,
                                                        const std::unordered_set<const NodeArg*>& graph_outputs,
                                                        const std::vector<const Node*>& group) const {
  std::unordered_set<const Node*> node_set;
  node_set.reserve(group.size());
  for (const Node* node : group) {
    node_set.insert(node);
  }

  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();

  std::unordered_set<const NodeArg*> node_outputs;
  std::unordered_set<const NodeArg*> subgraph_inputs;
  std::unordered_set<const NodeArg*> subgraph_outputs;
  std::vector<const NodeArg*> ordered_subgraph_inputs;
  std::vector<const NodeArg*> ordered_subgraph_outputs;

  for (const Node* node : group) {
    sub_graph->nodes.push_back(node->Index());

    for (const auto* input : node->InputDefs()) {
      // if the node input was not produced by this subgraph, add it to the subgraph inputs.
      if (node_outputs.count(input) == 0) {
        if (subgraph_inputs.count(input) == 0) {
          subgraph_inputs.insert(input);
          ordered_subgraph_inputs.push_back(input);
        }
      }
    }

    const auto& output_defs = node->OutputDefs();
    for (const auto* output_def : output_defs) {
      node_outputs.insert(output_def);
      // if output is overall graph output we need to produce it.
      if (graph_outputs.count(output_def) != 0) {
        ordered_subgraph_outputs.push_back(output_def);
      }
    }

    // if output connects to a node not in this subgraph we need to add it
    // unless it was already added as an overall graph output,
    for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
      if (node_set.count(&it->GetNode()) == 0) {
        const auto* output_def = output_defs[it->GetSrcArgIndex()];
        if (subgraph_outputs.count(output_def) == 0 && graph_outputs.count(output_def) == 0) {
          subgraph_outputs.insert(output_def);
          ordered_subgraph_outputs.push_back(output_def);
        }
      }
    }
  }

  // Assign inputs and outputs to subgraph's meta_def
  uint64_t model_hash;
  int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = "InternalTestingEP_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
  meta_def->domain = "InternalTesting";
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;

  for (const auto& input : ordered_subgraph_inputs) {
    meta_def->inputs.push_back(input->Name());
  }

  for (const auto& output : ordered_subgraph_outputs) {
    meta_def->outputs.push_back(output->Name());
  }

  sub_graph->SetMetaDef(std::move(meta_def));

  return std::make_unique<ComputeCapability>(std::move(sub_graph));
}

#ifndef NDEBUG
static void DiffSortOrders(const GraphViewer& graph_viewer,
                           const std::unordered_set<const Node*>& supported_nodes,
                           const std::vector<NodeIndex>& orig_order,
                           const std::vector<NodeIndex>& new_order,
                           bool debug_output) {
  auto indexes_str = [&](std::vector<NodeIndex>::const_iterator indexes_iter,
                         std::vector<NodeIndex>::const_iterator indexes_end) {
    std::ostringstream ss;
    bool currently_using_ep = false;
    while (indexes_iter != indexes_end) {
      NodeIndex index = *indexes_iter;
      const Node& node = *graph_viewer.GetNode(index);
      bool using_ep = supported_nodes.find(&node) != supported_nodes.end();
      if (using_ep != currently_using_ep) {
        ss << "\n"
           << (using_ep ? "YES: " : "NO: ");

        currently_using_ep = using_ep;
      }

      ss << graph_viewer.GetNode(index)->Name() << " ";

      ++indexes_iter;
    }

    DEBUG_LOGS(debug_output) << std::endl;

    return ss.str();
  };

  auto orig_iter = orig_order.cbegin();
  auto orig_end = orig_order.cend();
  auto new_iter = new_order.cbegin();
  auto new_end = new_order.cend();
  while (orig_iter != orig_end) {
    if (*orig_iter != *new_iter) {
      break;
    }

    ++orig_iter;
    ++new_iter;
  }

  if (orig_iter != orig_end) {
    DEBUG_LOGS(debug_output) << "Order differs from entry " << orig_iter - orig_order.cbegin() << "\n";
    DEBUG_LOGS(debug_output) << "Original order:" << indexes_str(orig_iter, orig_end) << "\n";
    DEBUG_LOGS(debug_output) << "Updated order:" << indexes_str(new_iter, new_end) << "\n";
  } else {
    DEBUG_LOGS(debug_output) << "No change in order from partition aware topo sort.\n";
  }
}
#endif

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

  // find any nodes we need to exclude
  auto excluded_nodes = CreateExcludedNodeSet(graph_viewer, stop_ops_);

  auto node_str = [](const Node* node) {
    std::ostringstream oss;
    oss << node->Index() << " '" << node->Name() << "'(" << node->OpType() << ")";
    return oss.str();
  };

  auto group_str = [&node_str](const std::vector<const Node*>& group) {
    const Node* start_node = group.front();
    const Node* end_node = group.back();
    std::ostringstream oss;
    oss << node_str(start_node) << " to " << node_str(end_node) << "\n";
    return oss.str();
  };

  auto cc_group_str = [&graph_viewer, &node_str](const ComputeCapability& group) {
    const Node* start_node = graph_viewer.GetNode(group.sub_graph->nodes.front());
    const Node* end_node = graph_viewer.GetNode(group.sub_graph->nodes.back());
    std::ostringstream oss;
    oss << node_str(start_node) << " to " << node_str(end_node) << "\n";
    return oss.str();
  };

  // partition aware sort. this groups all the nodes we can and can't handle
  const auto new_order = PartitionAwareTopoSort(graph_viewer, supported_nodes);

#ifndef NDEBUG
  if (debug_output_) {
    const auto& orig_order = graph_viewer.GetNodesInTopologicalOrder();
    DiffSortOrders(graph_viewer, supported_nodes, orig_order, new_order, debug_output_);
  }
#endif

  // create groups
  auto cur_topo_node = new_order.cbegin();
  auto end_topo_nodes = new_order.cend();
  std::queue<const Node*> nodes_to_process;       // supported nodes to process
  std::unordered_set<NodeIndex> processed_nodes;  // supported nodes we have processed

  // index of first node and vector of nodes. need vector as the entries must remain in topo order to figure out the
  // inputs and outputs for the IndexedSubGraph
  // TODO: If we don't need the merge logic we can simplify
  std::map<NodeIndex, std::vector<const Node*>> node_groups;
  std::vector<const Node*> cur_group;

  // ??? Do we need another loop here ???
  // We can add downstream nodes to be processed but need to check that all inputs for those nodes are available
  // before adding to the group. It may be the first time we see the downstream node that isn't the case.
  // For an input to be available, it must come from a node that we have fully processed
  // (not just be in the consideration set)
  bool check_excluded_nodes = !excluded_nodes.empty();
  const auto excluded_nodes_end = excluded_nodes.cend();

  while (cur_topo_node != end_topo_nodes) {
    NodeIndex node_index = *cur_topo_node;
    const Node* node = graph_viewer.GetNode(node_index);
    ++cur_topo_node;

    if (processed_nodes.find(node_index) == processed_nodes.cend()) {
      if (check_excluded_nodes && excluded_nodes.find(node) != excluded_nodes_end) {
        processed_nodes.insert(node->Index());
        continue;
      }

      bool supported = ops_.count(node->OpType()) != 0;
      bool in_partition = !cur_group.empty();

      // check if end of a partition.
      if (in_partition && !supported) {
        // TODO: Add as VERBOSE logging in debug build only
        DEBUG_LOGS(debug_output_) << "New partition due to " << node_str(node)
                                  << ". Nodes in old partition: " << cur_group.size() << "\n";
        DEBUG_LOGS(debug_output_) << group_str(cur_group) << "\n";

        node_groups.insert({cur_group.front()->Index(), std::move(cur_group)});
      }

      // add the node and any connected downstream nodes that we can handle if supported.
      // if not mark as processed so we know its inputs are available
      if (ops_.count(node->OpType())) {
        nodes_to_process.push(node);
      } else {
        processed_nodes.insert(node->Index());
      }

      while (!nodes_to_process.empty()) {
        node = nodes_to_process.front();
        nodes_to_process.pop();

        if (processed_nodes.find(node->Index()) == processed_nodes.cend()) {
          // add to partition if all inputs available
          bool inputs_available = std::all_of(
              node->InputNodesBegin(), node->InputNodesEnd(),
              [&processed_nodes](const Node& upstream_node) {
                return processed_nodes.find(upstream_node.Index()) != processed_nodes.cend();
              });

          if (inputs_available) {
            cur_group.push_back(node);
            processed_nodes.insert(node->Index());

            for (auto cur = node->OutputNodesBegin(), end = node->OutputNodesEnd(); cur != end; ++cur) {
              const Node& downstream_node = *cur;

              // nodes will get added to the queue once per input from a supported node.
              // we need this to happen as they can't be added to the group until all inputs are known to be available.
              if (ops_.count(downstream_node.OpType())) {
                nodes_to_process.push(&downstream_node);
              }
            }
          } else {
            // need another node providing input to be added to the group.
          }
        }
      };
    }

    //// ??? should we rely on the partition aware topo sort to stop, or should we identify the post-NMS nodes via edges?
    //// Using the edges seems safer but more expensive to do.
    //if (stop_at_nms_ && node->OpType() == "NonMaxSuppression") {
    //  break;
    //}
  }

  if (!cur_group.empty()) {
    node_groups.insert({cur_group.front()->Index(), std::move(cur_group)});
  }

  // create ComputeCapability instances
  const auto& graph_output_list = graph_viewer.GetOutputs();
  std::unordered_set<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());

  std::vector<std::unique_ptr<ComputeCapability>> results;
  results.reserve(node_groups.size());

  for (const auto& idx_to_group : node_groups) {
    results.push_back(MakeComputeCapability(graph_viewer, graph_outputs, idx_to_group.second));
  }

  return results;
}

std::vector<std::unique_ptr<ComputeCapability>>
InternalTestingExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                const std::vector<const KernelRegistry*>& /*registries*/) const {
  if (get_capability_version_ == 2) {
    ORT_THROW("DEAD");
  } else if (get_capability_version_ == 3) {
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

  const auto& graph_output_list = graph_viewer.GetOutputs();
  std::unordered_set<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());

  for (const auto& group : node_groups) {
    result.push_back(MakeComputeCapability(graph_viewer, graph_outputs, group));
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
