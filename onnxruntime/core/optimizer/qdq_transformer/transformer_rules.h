// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <string>
#include <utility>
#include <vector>

namespace onnxruntime {

class Graph;
class Node;

namespace QDQ {

enum class Direction { kInput,
                       kOutput };

// struct to define the location of an input or output definition for a Node
struct InOutDefSlot {
  Direction in_out;
  size_t idx;
};

// struct to define the input/output node, and
struct NodeAndArg {
  size_t idx;
  InOutDefSlot def_info;
};

struct NodeAttribute {
  size_t idx;
  std::string attribute_name;
};

struct NodeRequirements {
  std::unordered_set<std::string> op_types;

  // how do we do attribute requirements with different types?
};

struct ConstraintChecker {
  virtual bool operator()(const Graph&, const Node&) const { ORT_NOT_IMPLEMENTED(); };
};

// everything matches
struct NullChecker : public ConstraintChecker {
  bool operator()(const Graph&, const Node&) const override { return true; }
};

class OpTypeChecker : public ConstraintChecker {
 public:
  OpTypeChecker(const std::string& op_type) : op_type_{op_type} {}
  bool operator()(const Graph&, const Node& node) const override { return op_type_ == node.OpType(); }

 private:
  std::string op_type_;
};

// Base class for checking an input or output
// Will find the NodeArg for the given input/output and call the derived class to check it.
// If the input/output is missing, will return true if it is declared as being optional.
class InOutChecker : public ConstraintChecker {
 protected:
  InOutChecker(InOutDefSlot slot, bool optional = false) : slot_{slot}, optional_{optional} {}

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

 private:
  virtual bool Check(const Graph& graph, const NodeArg& arg) const = 0;
  const InOutDefSlot slot_;
  const bool optional_;
};

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

class ConstantInitializerChecker : public InOutChecker {
 public:
  ConstantInitializerChecker(InOutDefSlot slot) : InOutChecker{slot} {}

 private:
  bool Check(const Graph& graph, const NodeArg& node_arg) const override {
    return graph.GetConstantInitializer(node_arg.Name(), true) != nullptr;
  }
};

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

struct NodeSelectionRules {
  // node we look for before applying rules
  std::string target_op_;
  std::string target_domain_;

  // one entry per input to node. get node from edge. if all checkers return true, we're good
  // std::vector<std::vector<ConstraintChecker>> input_node_rules;
  //std::vector<std::vector<ConstraintChecker>> output_node_rules;
  // need to use unique_ptr, otherwise the real types get sliced and the polymorphism doesn't work.
  std::vector<std::unique_ptr<ConstraintChecker>> rules;
};

/*
// rules for moving edges involving replaced nodes
// movement of edges to downstream nodes of replaced nodes should be able to be automated
//  - need to track which new node is producing an output and move edges accordingly
//
// ??? Do we even need edge move rules? We really are moving where a NodeArg is consumed/produced not edges
//     Not sure if that makes a difference though...
struct EdgeMoveRule {
  NodeAndArg src_location;
  NodeAndArg dest_location;
};
*/

struct CreateNodeRule {
  std::string op_type;
  std::string name;  // generated?
  std::string domain;

  std::vector<NodeAndArg> input_defs;  // how do we manage values that are created as new initializers?
  std::vector<NodeAndArg> output_defs;

  std::map<std::string, NodeAttribute> attributes;

  //
};

// ??? Functor to specify inputs moving from other nodes
// Functor to create new input and return name
// Need input and output defs before creating node though
//   - can probably infer edge moves from this info

struct ActionRules {
  std::vector<CreateNodeRule> node_creation;
  // std::vector<EdgeMoveRule> edge_moves;  Edge moves should all be able to be inferred
  std::vector<size_t> node_removal;
};

struct Rules {
  NodeSelectionRules NodeSelection;
  ActionRules Actions;
};
}  // namespace QDQ
}  // namespace onnxruntime
