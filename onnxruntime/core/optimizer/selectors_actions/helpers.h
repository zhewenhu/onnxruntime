// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

//
// Selection helpers
//

// Group of nodes for processing. Accessors are provided for input/target/output nodes.
// A single variadic input OR output (not both - but no inferencing operator requires that) is currently supported
class NodesToOptimize {
 public:
  enum class NodeType {
    kInput,   // node providing input to target node
    kTarget,  // target node
    kOutput   // node consuming output from target node
  };

  struct NodeLocation {
    NodeType type;
    int index;
  };

  // index indexes in NodesToOptimize::Input and Output
  struct NodeIndexes {
    const std::vector<int>& input_node_indexes;
    bool include_target_node;
    const std::vector<int> output_node_indexes;
  };

  // nodes to assemble. num_inputs and num_outputs default to the size of input_nodes and output_nodes.
  // specify num_input_defs/num_output_defs if the last input/output is variadic
  NodesToOptimize(const std::vector<Node*>& input_nodes,
                  Node& target_node,
                  const std::vector<Node*>& output_nodes,
                  int num_input_defs = -1, int num_output_defs = -1);

  // construct from saved NodeIndex values. Use EmptyNodeIndex for nullptr entries in the vectors for missing optional
  // inputs
  NodesToOptimize(Graph& graph,
                  const std::vector<NodeIndex>& input_nodes,
                  NodeIndex target_node,
                  const std::vector<NodeIndex>& output_nodes,
                  int num_input_defs = -1, int num_output_defs = -1);

  static constexpr NodeIndex EmptyNodeIndex = std::numeric_limits<NodeIndex>::max();

  // number of inputs and outputs. these equate to the nodes providing an input/output (defined in the operator schema)
  // for the target node.
  // if the target node has a variadic input/output, the nodes providing those will always begin at the last entry
  // in the input/output nodes (i.e. at num_inputs - 1 or num_outputs - 1).
  //
  // e.g if there are 3 inputs (same applies to outputs)
  // if no variadic input
  //  num_inputs=3. optional inputs that are missing will have a nullptr entry
  // else variadic input
  //   if zero variadic values: num_inputs=3, last input is nullptr
  //   if one variadic value: num_inputs=3, last input is the single variadic input
  //   if multiple variadic values: num_inputs=3, total inputs = num_inputs + (NumVariadicInputs() - 1)
  const int num_inputs;
  const int num_outputs;

  bool HasVariadicInput() const { return num_extra_variadic_inputs_ > 0; }
  int NumVariadicInputs() const { return num_extra_variadic_inputs_ + 1; }

  bool HasVariadicOutput() const { return num_extra_variadic_outputs_ > 0; }
  int NumVariadicOutputs() const { return num_extra_variadic_outputs_ + 1; }

  // fetch an input.
  // valid indexes are 0 to num_inputs -1 if no variadic inputs.
  // if there are variadic inputs, valid indexes are 0 to num_inputs + num_extra_variadic_inputs - 1
  // e.g. 3 inputs. last is variadic with 3 values. num_inputs=3 num_extra_variadic_inputs=2 for a total of 5 inputs.
  Node* Input(int idx, bool required = true) const {
    return GetNode(idx, required);
  }

  // inputs filtered by index. includes all variadic.
  std::vector<Node*> Inputs(const std::vector<int>& indexes, bool required = true) const;

  Node* Target(bool required = true) const {
    return GetNode(0 + num_inputs + num_extra_variadic_inputs_, required);
  }

  Node* Output(int idx, bool required = true) const {
    return GetNode(idx + num_inputs + num_extra_variadic_inputs_ + 1, required);
  }

  // outputs filtered by index. includes all variadic.
  std::vector<Node*> Outputs(const std::vector<int>& indexes, bool required = true) const;

  // Get a single Node at a specific location.
  // Provided for efficiency over GetNodesAtLocation if you know that the input/output is not variadic.
  Node* GetNodeAtLocation(const NodeLocation& location, bool required = true) const;

  // Get the Node or Nodes at a specific index.
  // Enables generic Action implementations that support both single and variadic inputs/outputs.
  // Generally returns a single node unless it's a variadic input/output. Prefer GetNodeAtLocation if possible.
  std::vector<Node*> GetNodesAtLocation(const NodeLocation& location, bool required = true) const;

  gsl::span<Node* const> AllNodes() const { return gsl::make_span(nodes_); }

  NodesToOptimize(NodesToOptimize&&) = default;
  NodesToOptimize& operator=(NodesToOptimize&&) = default;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(NodesToOptimize);

 private:
  Node* GetNode(int index, bool required) const {
    Node* node = nullptr;
    ORT_ENFORCE(index < nodes_.size() && ((node = nodes_[index]) != nullptr || !required));

    return node;
  }

  // if last input is variadic, how many additional nodes are there for this input?
  // first one is included in num_inputs_
  int num_extra_variadic_inputs_{0};
  int num_extra_variadic_outputs_{0};
  std::vector<Node*> nodes_;
};

// Helper to build a NodesToOptimize instance
// Use in selector to incrementally add pieces
// Use in minimal build to convert saved node indexes to Node instances.
struct NodesToOptimizeBuilder {
  std::vector<Node*> input_nodes;
  Node* target_node{nullptr};
  std::vector<Node*> output_nodes;
  int num_input_defs{-1};
  int num_output_defs{-1};

  std::unique_ptr<NodesToOptimize> Build() {
    ORT_ENFORCE(target_node != nullptr, "A target node must be set.");
    return std::make_unique<NodesToOptimize>(input_nodes, *target_node, output_nodes, num_input_defs, num_output_defs);
  }
};

//
// Action helpers
//

enum class Direction { kInput,
                       kOutput };

// struct to define the location of an input or output definition for a Node
struct InOutDefSlot {
  Direction in_out;
  int idx;  // idx of -1 means 'all' if a source, or 'end' if a target
};

// TODO: Superceeded by ValueMoveInfo and NodeAndMoveInfo?
//struct NodeAndArg {
//  int node_idx;
//  InOutDefSlot in_out_slot;
//};

// Helper to define moving a value from one node to another
struct ValueMoveInfo {
  // simple 1:1 copy
  ValueMoveInfo(InOutDefSlot src_slot_in, InOutDefSlot dest_slot_in)
      : src_slot(src_slot_in), dest_slot(dest_slot_in) {}

  // copy all from source to destination
  ValueMoveInfo(Direction src_slot_type, Direction dest_slot_type, bool variadic = false)
      : src_slot{src_slot_type, -1},
        dest_slot{dest_slot_type, -1},
        copy_all{true},
        append{true},
        is_variadic{variadic} {
  }

  // append single value (may be variadic) from source to destination
  ValueMoveInfo(InOutDefSlot src_slot_in, Direction dest_slot_type, bool variadic = false)
      : src_slot(src_slot_in),
        dest_slot{dest_slot_type, -1},
        copy_all{false},
        append{true},
        is_variadic{variadic} {}

  InOutDefSlot src_slot;
  InOutDefSlot dest_slot;
  bool is_variadic{false};  // set if the copy involves a variadic
  bool copy_all{false};     // ignore src_slot.idx and copy all values
  bool append{false};       // ignore dest_slot.idx and append to existing values

 private:
  ValueMoveInfo() = default;
};

// info to move from an existing node to the target node
struct NodeAndMoveInfo {
  NodesToOptimize::NodeLocation src_node;
  ValueMoveInfo value_move_info;
};

//struct NodesAndMoveInfo {
//  NodesToOptimize::NodeLocation src_node;
//  NodesToOptimize::NodeLocation dest_node;
//  ValueMoveInfo value_move_info;
//};

// helper for moving inputs/outputs and their edges between nodes
struct MoveInputOutputHelper {
  static Status Move(Graph& graph, Node& src, Node& dest, const ValueMoveInfo& move_info) {
    return MoveInputOutputHelper(move_info).MoveImpl(graph, src, dest);
  }

 private:
  MoveInputOutputHelper(const ValueMoveInfo& move_info)
      : move_info_{move_info} {}

  Status MoveImpl(Graph& graph, Node& src, Node& dest) {
    ORT_RETURN_IF_ERROR(MoveNodeArg(graph, src, dest));
    MoveEdges(graph, src, dest);

    return Status::OK();
  }

  Status MoveNodeArg(Graph& graph, Node& src, Node& dest) const;

  // MoveEdges to find the matching edges
  void RemoveEdge(Graph& graph, Node& node, const InOutDefSlot& slot) const;

  void MoveEdges(Graph& graph, Node& src, Node& dest) const;

 private:
  const ValueMoveInfo& move_info_;
};

// check if a Node involved in the optimization can be safely removed.
// Requires the node_to_remove to not be producing any graph outputs, and to not be producing any outputs
// consumed by nodes other than the target node for the optimization.
bool CanSafelyRemoveNode(const Graph& graph, Node& node_to_remove, const Node* target_node);

}  // namespace onnxruntime
