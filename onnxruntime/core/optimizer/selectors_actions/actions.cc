// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/selectors_actions/actions.h"
#include "core/optimizer/selectors_actions/helpers.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status MoveInputOutput::operator()(Graph& graph, std::vector<Node*>& nodes) {
  Node& src = *GetNode(source_.node_idx, nodes);
  Node& dest = *GetNode(target_.node_idx, nodes);

  return MoveInputOutputHelper(source_.in_out_slot, target_.in_out_slot).Move(graph, src, dest);
}

Status MergeIntoExisting::operator()(Graph& graph, std::vector<Node*>& nodes) {
  for (const auto& src_dst : value_moves_) {
    // move inputs/output and associated edges
    ORT_RETURN_IF_ERROR(MoveInputOutput(src_dst.first, src_dst.second)(graph, nodes));
  }

  return RemoveNodes(nodes_to_remove_)(graph, nodes);
}

}  // namespace onnxruntime
