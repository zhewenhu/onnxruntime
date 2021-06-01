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

InternalTestingExecutionProvider::InternalTestingExecutionProvider(const std::unordered_set<std::string>& ops)
    : IExecutionProvider{utils::kInternalTestingExecutionProvider, true},
      ops_{ops} {
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

//struct NodeCompare {
//  const std::unordered_set<const Node*>& _supported_nodes;
//
//  NodeCompare(const std::unordered_set<const Node*>& supported_nodes) : _supported_nodes{supported_nodes} {
//  }
//
//  inline bool IsSupported(const Node* n) const {
//    return _supported_nodes.find(n) != _supported_nodes.cend();
//  }
//
//  bool operator()(const Node* n1, const Node* n2) const {
//    //bool n1_supported = IsSupported(n1);
//    //bool n2_supported = IsSupported(n2);
//
//    //if (n1_supported != n2_supported) {
//    //  return n2_supported;
//    //}
//
//    return n1->Index() < n2->Index();
//  }
//};

static std::vector<NodeIndex> PartitionAwareTopoSort(const GraphViewer& graph_viewer,
                                                     const std::unordered_set<const Node*>& supported_nodes) {
  std::set<const Node*, NodeCompare> to_visit;
  std::unordered_map<NodeIndex, size_t> in_degree;
  std::vector<NodeIndex> topo_order;

  auto num_nodes = graph_viewer.NumberOfNodes();
  topo_order.reserve(num_nodes);
  in_degree.reserve(num_nodes);

  // root nodes
  for (auto& node : graph_viewer.Nodes()) {
    size_t input_edge_count = node.GetInputEdgesCount();
    in_degree.insert({node.Index(), input_edge_count});
    if (input_edge_count == 0) {
      to_visit.insert(&node);
    }
  }

  auto is_supported = [&supported_nodes](const Node* n) {
    return supported_nodes.find(n) != supported_nodes.cend();
  };

  const Node* last = nullptr;
  // prefer unsupported nodes first. this will increase the number of inputs potentially available to partitions
  // handled by this EP.
  bool last_is_supported = false;

  while (!to_visit.empty()) {
    const Node* current = nullptr;

    // get next node. prefer a node on the same partition as the current node
    auto to_visit_end = to_visit.end();
    auto next_same = std::find_if(to_visit.begin(), to_visit_end,
                                  [last_is_supported, &is_supported](const Node* node) {
                                    return is_supported(node) == last_is_supported;
                                  });

    if (next_same != to_visit_end) {
      // found another node with the same partitioning
      current = *next_same;
      to_visit.erase(next_same);
    } else {
      current = *to_visit.begin();
      to_visit.erase(to_visit.begin());
      last_is_supported = !last_is_supported;
    }

    for (auto node_it = current->OutputNodesBegin(), end = current->OutputNodesEnd(); node_it != end; ++node_it) {
      in_degree[node_it->Index()]--;

      if (in_degree[node_it->Index()] == 0) {
        to_visit.insert(&*node_it);
      }
    }

    topo_order.push_back(current->Index());

    last = current;
  }

  if (graph_viewer.NumberOfNodes() != static_cast<int>(topo_order.size())) {
    ORT_THROW("LOGIC ERROR. Graph was valid when loaded but the topological sort has produced invalid output.");
  }

  return topo_order;
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

    // if output connects to a node not in this subgraph we need to produce it
    for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
      if (node_set.count(&it->GetNode()) == 0) {
        const auto* output_def = output_defs[it->GetSrcArgIndex()];
        if (subgraph_outputs.count(output_def) == 0) {
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

std::vector<std::unique_ptr<ComputeCapability>>
InternalTestingExecutionProvider::GetCapability2(const onnxruntime::GraphViewer& graph_viewer,
                                                 const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // Simple for experimentation - just use sets
  // std::vector<std::set<const Node*>> partitions;

  std::unordered_set<const Node*> supported_nodes;  // all nodes in graph we are handling

  // index of first node and vector of nodes. need vector as the entries must remain in topo order to figure out the
  // inputs and outputs for the IndexedSubGraph
  std::map<NodeIndex, std::vector<const Node*>> node_groups;
  std::vector<const Node*> cur_group;

  const auto& topo_nodes = graph_viewer.GetNodesInTopologicalOrder();
  auto cur_topo_node = topo_nodes.cbegin();
  auto end_topo_nodes = topo_nodes.cend();

  std::unordered_set<NodeIndex> processed_nodes;
  std::queue<const Node*> potentials;

  while (cur_topo_node != end_topo_nodes) {
    NodeIndex node_index = *cur_topo_node;

    if (processed_nodes.find(node_index) == processed_nodes.cend()) {
      const Node* node = graph_viewer.GetNode(node_index);

      // check if end of a partition.
      if (!cur_group.empty() && ops_.count(node->OpType()) == 0) {
        node_groups.insert({cur_group.front()->Index(), std::move(cur_group)});
      }

      // add the node and any connected downstream nodes that we can handle
      potentials.push(node);
      while (!potentials.empty()) {
        node = potentials.front();
        potentials.pop();

        if (processed_nodes.find(node_index) == processed_nodes.cend()) {
          // add to partition if supported
          if (ops_.count(node->OpType())) {
            cur_group.push_back(node);
            supported_nodes.insert(node);

            // add downstream nodes to potentials
            for (auto cur = node->OutputNodesBegin(), end = node->OutputNodesEnd(); cur != end; ++cur) {
              // TODO: Could use std::set so we don't process duplicates. But not sure how often we'd get them
              // so the cost of hashing on every insert may not outweigh a check of a duplicate entry that we have
              // processed already.
              potentials.push(&*cur);
            }
          }

          processed_nodes.insert(node->Index());
        }
      };
    }

    ++cur_topo_node;
  }

  if (!cur_group.empty()) {
    node_groups.insert({cur_group.front()->Index(), std::move(cur_group)});
  }

  if (node_groups.empty()) {
    return {};
  }

  // re-order using partitions as input
  const auto& orig_order = graph_viewer.GetNodesInTopologicalOrder();
  const auto new_order = PartitionAwareTopoSort(graph_viewer, supported_nodes);

#ifndef NDEBUG

  // Validate nodes in groups are unique
  for (const auto& idx_group_pair : node_groups) {
    std::unordered_set<const Node*> group_set;
    for (const Node* node : idx_group_pair.second) {
      bool inserted = group_set.insert(node).second;
      ORT_ENFORCE(inserted, "Duplicate entries in node group.");
    }
  }

  auto indexes_str = [&graph_viewer](const std::vector<NodeIndex>& node_indexes) {
    std::ostringstream ss;
    for (const NodeIndex index : node_indexes) {
      ss << graph_viewer.GetNode(index)->Name() << " ";
    }
    return ss.str();
  };

  std::cout << "Original order:" << indexes_str(orig_order) << "\n";
  std::cout << "Updated order:" << indexes_str(new_order) << "\n";

#endif

  const auto& graph_output_list = graph_viewer.GetOutputs();
  std::unordered_set<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());

  // create initial ComputeCapability
  std::unordered_map<NodeIndex, std::unique_ptr<ComputeCapability>> group_to_compute_capability;

  // inputs for a partition. key is the first node in the partition
  std::unordered_map<NodeIndex, std::unordered_set<const Node*>> group_to_input_nodes;

  for (const auto& idx_to_group : node_groups) {
    group_to_compute_capability.insert({idx_to_group.first,
                                        MakeComputeCapability(graph_viewer, graph_outputs, idx_to_group.second)});

    // find nodes this partition is dependent on by finding input edges to nodes external to the group
    std::unordered_set<const Node*> nodes_in_group(idx_to_group.second.begin(), idx_to_group.second.end());
    std::unordered_set<const Node*> input_nodes;

    for (const Node* node : idx_to_group.second) {
      for (auto input_node = node->InputNodesBegin(), end = node->InputNodesEnd(); input_node != end; ++input_node) {
        if (nodes_in_group.find(&*input_node) == nodes_in_group.cend()) {
          input_nodes.insert(&*input_node);
        }
      }
    }

    group_to_input_nodes.insert({idx_to_group.first, std::move(input_nodes)});
  }

  // merge partitions if possible
  //
  // iterate new topo sort
  // if node is first node of a partition...
  //   for all other later partitions (no entry in groups_seen yet from topo iteration)
  //     if input nodes of later partition have been seen (all in nodes_seen)
  //       merge the partitions
  std::unordered_set<NodeIndex> nodes_seen;
  std::unordered_set<NodeIndex> groups_seen;
  const auto group_input_nodes_end = group_to_input_nodes.cend();
  std::vector<std::unique_ptr<ComputeCapability>> result;

  for (const NodeIndex index : new_order) {
    // see if this node is the start of a group
    auto g1_iter = group_to_input_nodes.find(index);
    // check if node is start of a group && we haven't merged that group
    if (g1_iter != group_input_nodes_end && groups_seen.find(index) == groups_seen.cend()) {
      groups_seen.insert(index);

      auto g1_to_cc_iter = group_to_compute_capability.find(index);
      ORT_ENFORCE(g1_to_cc_iter != group_to_compute_capability.cend());
      std::unique_ptr<ComputeCapability>& g1_cc = g1_to_cc_iter->second;

      // check against other groups that we have not seen yet (i.e. they are downstream as we're iterating nodes
      // in the new topological order) to see if they can be merged
      for (const auto g2_iter : group_to_input_nodes) {
        if (groups_seen.find(g2_iter.first) == groups_seen.cend()) {
          // if all the input nodes for the second group have been seen their values are available and the two
          // groups can be merged
          const auto& g2_input_nodes = g2_iter.second;
          bool inputs_available = std::all_of(g2_input_nodes.cbegin(), g2_input_nodes.cend(),
                                              [&nodes_seen](const Node* input_node) {
                                                return nodes_seen.find(input_node->Index()) != nodes_seen.cend();
                                              });
          if (inputs_available) {
            // create new ComputeCapability from combined set of nodes.
            // We could be more efficient and implement a merge of the two ComputeCapability instances,
            //  but that is non-trivial code and as this happens during model load it's better to be simpler

            std::vector<const Node*>& g1_nodes = node_groups.find(index)->second;
            std::vector<const Node*>& g2_nodes = node_groups.find(g2_iter.first)->second;

            std::vector<const Node*> combined_nodes;
            combined_nodes.reserve(g1_nodes.size() + g2_nodes.size());

            std::copy(g1_nodes.begin(), g1_nodes.end(), std::back_inserter(combined_nodes));
            std::copy(g2_nodes.begin(), g2_nodes.end(), std::back_inserter(combined_nodes));

            // replace the old ComputeCapability with the combined one
            g1_cc = MakeComputeCapability(graph_viewer, graph_outputs, combined_nodes);

            // mark the merged group as seen so we ignore it
            groups_seen.insert(g2_iter.first);

            // and clear its entry from group_to_compute_capability so it can't be accidentally used
            // note: as we're currently iterating group_to_input_nodes it's not safe to erase from there
            auto g2_to_cc_iter = group_to_compute_capability.find(g2_iter.first);
            ORT_ENFORCE(g2_to_cc_iter != group_to_compute_capability.cend());
            group_to_compute_capability.erase(g2_iter.first);

            std::cout << "Merged group " << index << " with " << g2_iter.first << "\n";
          }
        }
      }

      // all merging is done so move the final value across to the results
      result.push_back(std::move(g1_cc));
    }
  }

  return result;
}

std::vector<std::unique_ptr<ComputeCapability>>
InternalTestingExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
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
    //  std::cout << "Fusing nodes: ";
    //  for (const auto& unfused_node : graph_viewer.Nodes()) {
    //    std::cout << " '" << unfused_node.Name() << "':" << unfused_node.Index();
    //  }
    //  std::cout << std::endl;
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
