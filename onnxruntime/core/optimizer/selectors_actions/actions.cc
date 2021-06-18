// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/selectors_actions/actions.h"
#include "core/optimizer/selectors_actions/helpers.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status RemoveNodes::operator()(Graph& graph, const NodesToOptimize& selected_nodes) const {
  auto remove = [&graph](Node* node, const Node* target_node = nullptr) {
    // TODO: Should we wire in the logger for messages about skipped nodes?
    if (node && CanSafelyRemoveNode(graph, *node, target_node)) {
      // TODO: It's slightly insane we don't support optionally removing the output edges as part of Graph::RemoveNode
      // but to make that change we need to validate a lot of existing code
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
    }
  };

  const Node* target_node = selected_nodes.Target();
  for (Node* node : selected_nodes.Inputs(nodes_to_remove_.input_node_indexes)) {
    remove(node, target_node);
  }

  // checking output edges in CanSafelyRemoveNode against the target node is no longer relevant
  // as we've processed all the inputs (so we can no longer have an output edge that points to it)
  target_node = nullptr;

  if (nodes_to_remove_.include_target_node) {
    remove(selected_nodes.Target());
  }

  for (Node* node : selected_nodes.Outputs(nodes_to_remove_.output_node_indexes)) {
    remove(node);
  }

  return Status::OK();
}

Status RemoveAllNodes::operator()(Graph& graph, const NodesToOptimize& selected_nodes) const {
  const Node* target_node = selected_nodes.Target();

  for (auto* node : selected_nodes.AllNodes()) {
    if (node != nullptr) {
      // AllNodes are ordered as inputs, target, output. if we just removed the target node, it's no longer valid
      if (node == target_node) {
        target_node = nullptr;
      }

      if (CanSafelyRemoveNode(graph, *node, target_node)) {
        graph_utils::RemoveNodeOutputEdges(graph, *node);
        graph.RemoveNode(node->Index());
      }
    }
  }

  return Status::OK();
}

Status MoveInputOutput::operator()(Graph& graph, const NodesToOptimize& selected_nodes) const {
  std::vector<Node*> src_nodes{nullptr};

  // these calls currently require the requested nodes to exist
  if (!move_info_.value_move_info.is_variadic) {
    // common case first. just set value in src_nodes (avoids cost of creating a new vector)
    src_nodes[0] = selected_nodes.GetNodeAtLocation(move_info_.src_node);
  } else {
    src_nodes = selected_nodes.GetNodesAtLocation(move_info_.src_node);
  }

  Node& dest_node = *selected_nodes.Target();

  for (Node* src_node : src_nodes) {
    if (src_node != nullptr) {
      ORT_RETURN_IF_ERROR(MoveInputOutputHelper::Move(graph, *src_node, dest_node, move_info_.value_move_info));
    }
  }

  return Status::OK();
}

Status MergeIntoExisting::operator()(Graph& graph, const NodesToOptimize& selected_nodes) const {
  // needs NodeAndMoveInfo
  for (const auto& value_mover : value_movers_) {
    // move inputs/output and associated edges
    ORT_RETURN_IF_ERROR(value_mover(graph, selected_nodes));
  }

  return node_remover_(graph, selected_nodes);
}

}  // namespace onnxruntime
