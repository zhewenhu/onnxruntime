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
}  // namespace

Status MoveInputOutputHelper::MoveNodeArg(Graph& graph, Node& src, Node& dest) const {
  auto& src_defs = (src_slot_.in_out == Direction::kInput)
                       ? src.MutableInputDefs()
                       : src.MutableOutputDefs();

  auto& dest_defs = (dest_slot_.in_out == Direction::kInput)
                        ? dest.MutableInputDefs()
                        : dest.MutableOutputDefs();

  auto process = [&](int src_idx) {
    ORT_ENFORCE((src_idx == -1 || src_idx < src_defs.size()) &&
                    (dest_slot_.idx == -1 || dest_slot_.idx < dest_defs.size()),
                "Index out of range");

    if (append_) {
      dest_defs.push_back(src_defs[src_idx]);
      if (dest_slot_.in_out == Direction::kInput) {
        // TODO: If we need to support variadic inputs appending 1 each time won't work
        dest.MutableInputArgsCount().push_back(1);
      }
    } else {
      // remove any edge to the slot we're replacing
      RemoveEdge(graph, dest, dest_slot_);
      dest_defs[dest_slot_.idx] = src_defs[src_slot_.idx];
    }
  };

  if (copy_all_) {
    for (int i = 0, end = gsl::narrow<int>(src_defs.size()); i < end; ++i) {
      process(i);
    }
  } else {
    process(src_slot_.idx);
  }

  return Status::OK();
}

// MoveEdges to find the matching edges
void MoveInputOutputHelper::RemoveEdge(Graph& graph, Node& node, const InOutDefSlot& slot) const {
  ProcessEdge(graph, node, slot, nullptr, nullptr);
}

void MoveInputOutputHelper::MoveEdges(Graph& graph, Node& src, Node& dest) const {
  ProcessEdge(graph, src, src_slot_, &dest, &dest_slot_);
}

}  // namespace onnxruntime
