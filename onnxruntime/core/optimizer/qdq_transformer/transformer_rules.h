// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

class Graph;
class Node;

namespace QDQ {

enum class Direction { kInput,
                       kOutput };

// struct to define the location of an input or output definition for a Node
struct InOutDefSlot {
  Direction in_out;
  int idx;  // idx of -1 means 'all' if a source, or 'end' if a target
};

//const Node* NodeFromSlot(const Node& node, const InOutDefSlot& slot);

struct NodeAndArg {
  size_t node_idx;
  InOutDefSlot in_out_slot;
};

struct ConstraintChecker {
  virtual bool operator()(const Graph& graph, const Node& node) const = 0;
  virtual ~ConstraintChecker() = default;
};

// Base class for checking an input or output
// Will find the NodeArg for the given input/output and call the derived class to check it.
// If the input/output is missing, will return true if it is declared as being optional.
class InOutChecker : public ConstraintChecker {
 public:
  bool operator()(const Graph& graph, const Node& node) const override {
    const auto& node_arg = slot_.in_out == Direction::kInput
                               ? *node.InputDefs().at(slot_.idx)
                               : *node.OutputDefs().at(slot_.idx);

    if (node_arg.Exists()) {
      return Check(graph, node_arg);
    } else {
      return optional_;
    }
  }

 protected:
  InOutChecker(InOutDefSlot slot, bool optional = false) : slot_{slot}, optional_{optional} {}
  const InOutDefSlot& Slot() const { return slot_; }

 private:
  // method for derived classes to implement
  virtual bool Check(const Graph& graph, const NodeArg& arg) const = 0;

  const InOutDefSlot slot_;
  const bool optional_;
};

class IsScalarChecker : public InOutChecker {
 public:
  IsScalarChecker(InOutDefSlot slot) : InOutChecker{slot} {}

 private:
  bool Check(const Graph&, const NodeArg& node_arg) const override {
    return optimizer_utils::IsScalar(node_arg);
  }
};

class ConstantInitializerChecker : public InOutChecker {
 public:
  ConstantInitializerChecker(InOutDefSlot slot) : InOutChecker{slot} {}

 private:
  bool Check(const Graph& graph, const NodeArg& node_arg) const override {
    return graph.GetConstantInitializer(node_arg.Name(), true) != nullptr;
  }
};

class ConstantScalarChecker : public InOutChecker {
 public:
  ConstantScalarChecker(InOutDefSlot slot) : InOutChecker{slot} {}

 private:
  bool Check(const Graph& graph, const NodeArg& node_arg) const override {
    return optimizer_utils::IsScalar(node_arg) &&
           graph.GetConstantInitializer(node_arg.Name(), true) != nullptr;
  }
};

/*
// check that an input or output is a tensor with the specified element type
class TensorElemTypeChecker : public InOutChecker {
 public:
  TensorElemTypeChecker(InOutDefSlot slot, int32_t type) : InOutChecker{slot}, elem_type_{type} {}

 private:
  bool Check(const Graph&, const NodeArg& node_arg) const override {
    const auto* type = node_arg.TypeAsProto();
    return type != nullptr &&
           type->has_tensor_type() &&
           type->tensor_type().elem_type() == elem_type_;
  }

  const int32_t elem_type_;
};

// check the input or output count of a node
// does not count missing optional inputs
class InOutCount : public ConstraintChecker {
 public:
  InOutCount(Direction type, size_t count) : type_{type}, count_{count} {}

  bool operator()(const Graph&, const Node& node) const override {
    const auto& defs = type_ == Direction::kInput ? node.InputDefs() : node.OutputDefs();
    auto num_defs = gsl::narrow_cast<size_t>(std::count_if(defs.cbegin(), defs.cend(),
                                                           [](const NodeArg* def) { return def->Exists(); }));

    return num_defs == count_;
  }

 private:
  const Direction type_;
  const size_t count_;
};

// check if an input or output is coming from a specific OpType
class InOutOpTypeChecker : public ConstraintChecker {
 public:
  InOutOpTypeChecker(InOutDefSlot slot, const std::string& op_type) : slot_{slot}, op_type_{op_type} {
  }

  bool operator()(const Graph&, const Node& node) const override {
    const Node* in_out_node = NodeFromSlot(node, slot_);
    return in_out_node && in_out_node->OpType() == op_type_;

    if (slot_.in_out == Direction::kInput) {
      auto iter = std::find_if(node.InputEdgesBegin(), node.InputEdgesEnd(),
                               [this](const Node::EdgeEnd& edge) {
                                 return (edge.GetDstArgIndex() == slot_.idx);
                               });

      return iter != node.InputEdgesEnd() &&
             iter->GetNode().OpType() == op_type_;
    } else {
      auto iter = std::find_if(node.OutputEdgesBegin(), node.OutputEdgesEnd(),
                               [this](const Node::EdgeEnd& edge) {
                                 return (edge.GetSrcArgIndex() == slot_.idx);
                               });

      return iter != node.OutputEdgesEnd() &&
             iter->GetNode().OpType() == op_type_;
    }
  }

 private:
  InOutDefSlot slot_;
  const std::string op_type_;
};
*/

struct NodeSelector {
  // Select one or more nodes for an Action to process if the constraints are satisfied
  virtual bool operator()(Graph& graph, const Node& node, std::vector<Node*>& selection) const = 0;
  virtual ~NodeSelector() = default;
};

// Base QDQ checker. Provides the DQ and Q nodes to the operator specific checkers
class QDQSelector : public NodeSelector {
 public:
  bool operator()(Graph& graph, const Node& node, std::vector<Node*>& selection) const override;

 protected:
  QDQSelector() = default;

  // override if you need to add entries for missing optional DQ inputs
  // Called post-Check if Check returned `true`
  virtual void AdjustDQNodes(std::vector<const Node*>& /*dq_nodes*/) const {}

  // base check that we have the expected number of QDQ inputs/outputs, and `node` isn't producing a graph output
  // num_dq_inputs defaults to the number of inputs `node` has
  bool CheckQDQNodes(const Graph& graph, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes,
                     size_t num_dq_inputs = std::numeric_limits<size_t>::max()) const {
    if (num_dq_inputs == std::numeric_limits<size_t>::max()) {
      const auto& input_defs = node.InputDefs();
      // adjust for an optional input that has an entry but does not exist
      num_dq_inputs = std::count_if(input_defs.cbegin(), input_defs.cend(),
                                    [](const NodeArg* def) { return def && def->Exists(); });
    }

    return num_dq_inputs == dq_nodes.size() &&
           node.OutputDefs().size() == q_nodes.size() &&
           optimizer_utils::CheckOutputEdges(graph, node, q_nodes.size());
  }

 private:
  bool virtual Check(const Graph& graph, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes) const = 0;
};

// Single DQ -> node -> Q.
// Zero point and scale are constant scalars and match
class QDQSimpleSelector : public QDQSelector {
 public:
  QDQSimpleSelector();

 private:
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  const ConstantScalarChecker dq_scale_is_constant_scalar_;
  const ConstantScalarChecker dq_zero_point_is_constant_scalar_;
  const ConstantScalarChecker q_scale_is_constant_scalar_;
  const ConstantScalarChecker q_zero_point_is_constant_scalar_;
};

// 2 DQ nodes providing input -> node -> Q
class QDQBinarySelector : public QDQSelector {
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

class QDQConvSelector : public QDQSelector {
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  void AdjustDQNodes(std::vector<const Node*>& dq_nodes) const override;
};

//
// Actions
//

// actions that are applied to a set of nodes identified during selection
struct Action {
  virtual Status operator()(Graph&, std::vector<Node*>& nodes) = 0;
  virtual ~Action() = default;

  // helper to validate the index when fetching a node
  Node* GetNode(size_t index, const std::vector<Node*>& nodes, bool required = true) {
    Node* node = nullptr;
    ORT_ENFORCE(index < nodes.size() &&
                ((node = nodes[index]) != nullptr || !required));

    return node;
  }
};

// Move a value from one node to another and adjusts edges accordingly
struct MoveInputOutput : public Action {
  MoveInputOutput(NodeAndArg src, NodeAndArg dest) : source_{src}, target_{dest} {
  }

  Status operator()(Graph& graph, std::vector<Node*>& nodes) override;

 private:
  NodeAndArg source_;
  NodeAndArg target_;
};

struct RemoveNodes : public Action {
  RemoveNodes(const std::vector<size_t>& node_indexes)
      : nodes_to_remove_{node_indexes} {}

  Status operator()(Graph& graph, std::vector<Node*>& nodes) override {
    for (auto idx : nodes_to_remove_) {
      Node& node = *GetNode(idx, nodes);
      graph_utils::RemoveNodeOutputEdges(graph, node);
      graph.RemoveNode(node.Index());
    }

    return Status::OK();
  }

 private:
  const std::vector<size_t> nodes_to_remove_;
};

struct RemoveAllNodes : public Action {
  Status operator()(Graph& graph, std::vector<Node*>& nodes) override {
    for (auto* node : nodes) {
      if (node != nullptr) {
        graph_utils::RemoveNodeOutputEdges(graph, *node);
        graph.RemoveNode(node->Index());
      }
    }

    return Status::OK();
  }
};

struct MergeIntoExisting : public Action {
  MergeIntoExisting(std::initializer_list<std::pair<NodeAndArg, NodeAndArg>> value_moves,
                    std::initializer_list<size_t> nodes_to_remove)
      : value_moves_{value_moves},
        nodes_to_remove_{nodes_to_remove} {
  }

 private:
  virtual Status operator()(Graph&, std::vector<Node*>& nodes);

  std::vector<std::pair<NodeAndArg, NodeAndArg>> value_moves_;

  std::vector<size_t> nodes_to_remove_;  // index into 'nodes' vector that operator() is called with
};

struct SetOptionalZeroPoint : public Action {
  SetOptionalZeroPoint(std::initializer_list<size_t> nodes_to_update)
      : nodes_to_update_{nodes_to_update} {}

 private:
  virtual Status operator()(Graph&, std::vector<Node*>& nodes);

  std::vector<size_t> nodes_to_update_;  // index into 'nodes' vector that operator() is called with

  static const ONNX_NAMESPACE::TensorProto optional_zero_point_int8_;
  static const ONNX_NAMESPACE::TensorProto optional_zero_point_uint8_;
};

struct ReplaceWithNew : public Action {
  ReplaceWithNew(size_t node_to_replace,
                 const std::string& domain,
                 std::initializer_list<std::pair<NodeAndArg, InOutDefSlot>> value_moves);

 private:
  Status operator()(Graph&, std::vector<Node*>& nodes) override;

  // TODO: setup mechanism to create a new NodeArg
  // If we use resize on the input defs we can do the moves and directly populate the slot with a new NodeArg
  // but this may not be needed for QDQ.

  size_t node_to_replace_;
  const std::string domain_;
  std::vector<std::pair<NodeAndArg, InOutDefSlot>> value_moves_;
};

struct SelectorAndActions {
  // Operator and supported versions for the node that selection will start from.
  std::unordered_map<std::string, std::vector<int>> ops_and_versions;
  std::unique_ptr<NodeSelector> selector;
  std::vector<std::unique_ptr<Action>> actions;
};

}  // namespace QDQ
}  // namespace onnxruntime
