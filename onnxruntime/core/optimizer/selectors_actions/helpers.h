// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

enum class Direction { kInput,
                       kOutput };

// struct to define the location of an input or output definition for a Node
struct InOutDefSlot {
  Direction in_out;
  int idx;  // idx of -1 means 'all' if a source, or 'end' if a target
};

struct NodeAndArg {
  size_t node_idx;
  InOutDefSlot in_out_slot;
};

// helper for moving inputs/outputs and their edges between nodes
struct MoveInputOutputHelper {
  MoveInputOutputHelper(const InOutDefSlot& src_slot, const InOutDefSlot& dest_slot)
      : src_slot_{src_slot},
        dest_slot_{dest_slot},
        copy_all_{src_slot.idx == -1},
        append_{dest_slot.idx == -1} {
    // if copying all you have to append
    assert(copy_all_ == false || append_ == true);
  }

  Status Move(Graph& graph, Node& src, Node& dest) {
    ORT_RETURN_IF_ERROR(MoveNodeArg(graph, src, dest));
    MoveEdges(graph, src, dest);

    return Status::OK();
  }

 private:
  Status MoveNodeArg(Graph& graph, Node& src, Node& dest) const;

  // MoveEdges to find the matching edges
  void RemoveEdge(Graph& graph, Node& node, const InOutDefSlot& slot) const;

  void MoveEdges(Graph& graph, Node& src, Node& dest) const;

 private:
  InOutDefSlot src_slot_;
  InOutDefSlot dest_slot_;
  bool copy_all_;  // copy all the input/output definitions from the source to the target
  bool append_;    // append the input/output definition to the target
};

}  // namespace onnxruntime
