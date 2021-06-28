// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/selectors_actions/actions.h"
#include "core/optimizer/selectors_actions/helpers.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {

// Check if a node involved in an optimization can be safely removed.
// This requires that the optimizer correctly handles nodes producing graph outputs and does not attempt to delete
// one of those nodes unless it has created a new source for the graph output. As we can't easily remove the NodeArg
// from the Node::OutputDefs for the node being removed, we do not check if the node provides graph outputs here.
bool CanSafelyRemoveNode(Node& node_to_remove, const std::unordered_set<const Node*>& removal_set) {
  bool safe = true;
  for (auto iter = node_to_remove.OutputEdgesBegin(), end = node_to_remove.OutputEdgesEnd(); iter != end; ++iter) {
    if (removal_set.find(&iter->GetNode()) == removal_set.cend()) {
      safe = false;
      break;
    }
  }

  return safe;
}

// remove nodes if it is safe to do so. 'safe' means no output edges to nodes not in the set of nodes being removed.
// there is NO check on a node producing a graph output. we assume the optimizer has handled that already.
void SafelyRemoveNodes(Graph& graph, const std::vector<Node*>& nodes_to_remove, const Node* skip_target) {
  std::unordered_set<const Node*> removal_set(nodes_to_remove.cbegin(), nodes_to_remove.cend());

  for (Node* node : nodes_to_remove) {
    if (node && node != skip_target && CanSafelyRemoveNode(*node, removal_set)) {
      // TODO: It's slightly insane we don't support optionally removing the output edges as part of Graph::RemoveNode
      // but to make that change we need to validate a lot of existing code
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
    }
  }
}
}  // namespace

Status RemoveNodes::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  Node* skip_target = preserve_target_node_ ? &selected_nodes.Target() : nullptr;
  SafelyRemoveNodes(graph, selected_nodes.AllNodes(), skip_target);

  return Status::OK();
}

Status MergeIntoTarget::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  // sanity check. any incorrect usage would happen during development so no need to use ORT_ENFORCE
  assert(selected_nodes.num_inputs <= 1 && selected_nodes.num_outputs <= 1 &&
         !selected_nodes.HasVariadicInput() && !selected_nodes.HasVariadicOutput());

  if (selected_nodes.num_inputs == 1) {
    // move inputs from the input node to the target node as the input node will be deleted
    ORT_RETURN_IF_ERROR(MoveInputOutput(graph, *selected_nodes.Input(0), selected_nodes.Target(),
                                        {ArgType::kInput, ArgType::kInput}));
  }

  if (selected_nodes.num_outputs == 1) {
    // move outputs from the output node to the target node as the output node will be deleted
    ORT_RETURN_IF_ERROR(MoveInputOutput(graph, *selected_nodes.Output(0), selected_nodes.Target(),
                                        {ArgType::kOutput, ArgType::kOutput}));
  }

  return node_remover_.Run(graph, selected_nodes);
}

ReplaceWithNew::ReplaceWithNew(const std::string& domain,
                               const std::string& op_name,
                               std::vector<NodeAndMoveInfo>&& value_moves)
    : domain_{domain}, op_{op_name}, value_moves_{std ::move(value_moves)} {
}

Status ReplaceWithNew::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  auto& target = selected_nodes.Target();

  std::string op_type = OpType(selected_nodes);

  // create node. we'll populate the input and output defs via moves
  auto& replacement = graph.AddNode(target.Name(),
                                    op_type,
                                    target.Description(),
                                    {},  // input defs
                                    {},  // output defs
                                    &target.GetAttributes(),
                                    domain_);

  replacement.SetExecutionProviderType(kCpuExecutionProvider);

  ORT_RETURN_IF_ERROR(MoveInputOutput(graph, selected_nodes, replacement, value_moves_));
  return node_remover_.Run(graph, selected_nodes);
}

}  // namespace onnxruntime
