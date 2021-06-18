// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/selectors_actions/actions.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {
// remove edges for the src+src_slot if dest+dest_slot not provided.
// moves edges from src+src_slot to dest node+dest_slot if provided.
static void ProcessEdge(Graph& graph, Node& src, const InOutDefSlot& src_slot,
                        Node* dest, const InOutDefSlot* dest_slot) {
  if (src_slot.in_out == Direction::kInput) {
    // move input edge if present
    auto iter = std::find_if(src.InputEdgesBegin(), src.InputEdgesEnd(),
                             [&src_slot](const Node::EdgeEnd& edge) {
                               return (edge.GetDstArgIndex() == src_slot.idx);
                             });

    // initializer or graph input doesn't have an edge so either zero or one edges to process
    if (iter != src.InputEdgesEnd()) {
      const Node& iter_node = iter->GetNode();
      graph.RemoveEdge(iter_node.Index(), src.Index(), iter->GetSrcArgIndex(), src_slot.idx);
      if (dest && dest_slot) {
        graph.AddEdge(iter_node.Index(), dest->Index(), iter->GetSrcArgIndex(), dest_slot->idx);
      }
    }

  } else {
    // otherwise we need to move all output edges (if any)
    auto edges = graph_utils::GraphEdge::GetNodeOutputEdges(src, src_slot.idx);
    graph_utils::GraphEdge::RemoveGraphEdges(graph, edges);
    if (dest && dest_slot) {
      for (const auto& edge : edges) {
        graph.AddEdge(dest->Index(), edge.dst_node, dest_slot->idx, edge.dst_arg_index);
      }
    }
  }
}

Node* GetNodeByNodeIndex(Graph& graph, NodeIndex idx, bool required = true) {
  if (idx == NodesToOptimize::EmptyNodeIndex) {
    return nullptr;
  }

  Node* node = graph.GetNode(idx);
  ORT_ENFORCE(node != nullptr || !required, "Node was required but not found at index ", idx);

  return node;
}

std::vector<Node*> GetNodesByNodeIndex(Graph& graph, const std::vector<NodeIndex>& indexes) {
  std::vector<Node*> nodes;
  nodes.reserve(indexes.size());
  std::for_each(indexes.cbegin(), indexes.cend(),
                [&graph, &nodes](NodeIndex idx) {
                  nodes.push_back(GetNodeByNodeIndex(graph, idx));
                });

  return nodes;
}
}  // namespace

//
// Selections
//

NodesToOptimize::NodesToOptimize(const std::vector<Node*>& input_nodes,
                                 Node& target_node,
                                 const std::vector<Node*>& output_nodes,
                                 int num_input_defs, int num_output_defs)
    : num_inputs{num_input_defs == -1 ? gsl::narrow_cast<int>(input_nodes.size()) : num_input_defs},
      num_outputs{num_output_defs == -1 ? gsl::narrow_cast<int>(output_nodes.size()) : num_output_defs} {
  //
  if (num_input_defs != -1) {
    num_extra_variadic_inputs_ = gsl::narrow_cast<int>(input_nodes.size()) - num_input_defs;
  }

  if (num_output_defs != -1) {
    num_extra_variadic_outputs_ = gsl::narrow_cast<int>(output_nodes.size()) - num_output_defs;
  }

  nodes_.reserve(num_inputs + num_extra_variadic_inputs_ + 1 + num_outputs + num_extra_variadic_outputs_);
  std::copy(input_nodes.begin(), input_nodes.end(), std::back_inserter(nodes_));
  nodes_.push_back(&target_node);
  std::copy(output_nodes.begin(), output_nodes.end(), std::back_inserter(nodes_));
}

NodesToOptimize::NodesToOptimize(Graph& graph,
                                 const std::vector<NodeIndex>& input_nodes,
                                 NodeIndex target_node,
                                 const std::vector<NodeIndex>& output_nodes,
                                 int num_input_defs, int num_output_defs)
    : NodesToOptimize{GetNodesByNodeIndex(graph, input_nodes),
                      *GetNodeByNodeIndex(graph, target_node),
                      GetNodesByNodeIndex(graph, output_nodes),
                      num_input_defs,
                      num_output_defs} {
}

std::vector<Node*> NodesToOptimize::Inputs(const std::vector<int>& indexes, bool required) const {
  std::vector<Node*> results;
  results.reserve(num_inputs + num_extra_variadic_inputs_);

  for (auto idx : indexes) {
    if (idx == num_inputs - 1 && HasVariadicInput()) {
      for (int i = 0, end = NumVariadicInputs(); i < end; ++i) {
        results.push_back(GetNode(idx + i, required));
      }
    } else {
      results.push_back(GetNode(idx, required));
    }
  }

  return results;
}

std::vector<Node*> NodesToOptimize::Outputs(const std::vector<int>& indexes, bool required) const {
  std::vector<Node*> results;
  results.reserve(num_outputs + num_extra_variadic_outputs_);

  // offset by all the inputs and the target node
  const int offset = num_inputs + num_extra_variadic_inputs_ + 1;

  for (auto idx : indexes) {
    if (idx == num_outputs - 1 && HasVariadicOutput()) {
      for (int i = 0, end = NumVariadicOutputs(); i < end; ++i) {
        results.push_back(GetNode(offset + idx + i, required));
      }
    } else {
      results.push_back(GetNode(offset + idx, required));
    }
  }

  return results;
}

Node* NodesToOptimize::GetNodeAtLocation(const NodeLocation& location, bool required) const {
  if (location.type == NodeType::kInput) {
    return Input({location.index}, required);
  } else if (location.type == NodeType::kOutput) {
    return Output({location.index}, required);
  } else
    return {Target()};
};

std::vector<Node*> NodesToOptimize::GetNodesAtLocation(const NodeLocation& location, bool required) const {
  if (location.type == NodeType::kInput) {
    return Inputs({location.index}, required);
  } else if (location.type == NodeType::kOutput) {
    return Outputs({location.index}, required);
  } else
    return {Target()};
};

//
//
// Actions
//
Status MoveInputOutputHelper::MoveNodeArg(Graph& graph, Node& src, Node& dest) const {
  auto& src_defs = (move_info_.src_slot.in_out == Direction::kInput)
                       ? src.MutableInputDefs()
                       : src.MutableOutputDefs();

  auto& dest_defs = (move_info_.dest_slot.in_out == Direction::kInput)
                        ? dest.MutableInputDefs()
                        : dest.MutableOutputDefs();

  auto process = [&](int src_idx) {
    ORT_ENFORCE((move_info_.copy_all || static_cast<size_t>(src_idx) < src_defs.size()) &&
                    (move_info_.append || static_cast<size_t>(move_info_.dest_slot.idx) < dest_defs.size()),
                "Index out of range");

    if (move_info_.append) {
      dest_defs.push_back(src_defs[src_idx]);
      if (move_info_.dest_slot.in_out == Direction::kInput) {
        // TODO: If we need to support variadic inputs appending 1 each time won't work
        dest.MutableInputArgsCount().push_back(1);
      }
    } else {
      // remove any edge to the slot we're replacing
      RemoveEdge(graph, dest, move_info_.dest_slot);
      dest_defs[move_info_.dest_slot.idx] = src_defs[move_info_.src_slot.idx];
    }
  };

  if (move_info_.copy_all) {
    for (int i = 0, end = gsl::narrow<int>(src_defs.size()); i < end; ++i) {
      process(i);
    }
  } else {
    process(move_info_.src_slot.idx);
  }

  return Status::OK();
}

// MoveEdges to find the matching edges
void MoveInputOutputHelper::RemoveEdge(Graph& graph, Node& node, const InOutDefSlot& slot) const {
  ProcessEdge(graph, node, slot, nullptr, nullptr);
}

void MoveInputOutputHelper::MoveEdges(Graph& graph, Node& src, Node& dest) const {
  ProcessEdge(graph, src, move_info_.src_slot, &dest, &move_info_.dest_slot);
}

bool CanSafelyRemoveNode(const Graph& graph, Node& node_to_remove, const Node* target_node) {
  bool safe = true;
  if (target_node != nullptr) {
    // TODO: Refine this as it's a little vague.
    // Caller should provide a bool saying whether we need to check graph outputs
    // Assumption is that once we get to the downstream nodes from the target nodes, we wouldn't have touched those
    // nodes if they were producing graph outputs. Inputs to the target node are a different story, as it's fine
    // to take their outputs and use it in a replacement node
    if (graph.GetNodeProvidesGraphOutput(node_to_remove)) {
      return false;
    }

    for (auto iter = node_to_remove.OutputEdgesBegin(), end = node_to_remove.OutputEdgesEnd(); iter != end; ++iter) {
      if (&iter->GetNode() != target_node) {
        return false;
      }
    }
  } else {
    safe = node_to_remove.GetOutputEdgesCount() == 0;
  }

  return safe;
}

}  // namespace onnxruntime
