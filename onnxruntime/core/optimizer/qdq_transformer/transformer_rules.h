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
  size_t idx;
};

//// target of an operation defined with TargetNode combined with an InOutDefSlot
//// If the
//enum class TargetNode { kInput,
//                        kOutput,
//                        kExisting,
//                        kNew };

const Node* NodeFromSlot(const Node& node, const InOutDefSlot& slot);

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

//class InOutOpTypeChecker : public ConstraintChecker {
// public:
//  InOutOpTypeChecker(InOutDefSlot slot, const std::string& op_type) : slot_{slot}, op_type_{op_type} {
//  }
//
//  bool operator()(const Graph& graph, const Node& node) const override {
//    const Node* in_out_node = NodeFromSlot(node, slot_);
//    return in_out_node && in_out_node->OpType() == op_type_;
//
//    //if (slot_.in_out == Direction::kInput) {
//    //  auto iter = std::find_if(node.InputEdgesBegin(), node.InputEdgesEnd(),
//    //                           [this](const Node::EdgeEnd& edge) {
//    //                             return (edge.GetDstArgIndex() == slot_.idx);
//    //                           });
//
//    //  return iter != node.InputEdgesEnd() &&
//    //         iter->GetNode().OpType() == op_type_;
//    //} else {
//    //  auto iter = std::find_if(node.OutputEdgesBegin(), node.OutputEdgesEnd(),
//    //                           [this](const Node::EdgeEnd& edge) {
//    //                             return (edge.GetSrcArgIndex() == slot_.idx);
//    //                           });
//
//    //  return iter != node.OutputEdgesEnd() &&
//    //         iter->GetNode().OpType() == op_type_;
//    //}
//  }
//
// private:
//  InOutDefSlot slot_;
//  const std::string op_type_;
//};

// Base QDQ checker. Provides the DQ and Q nodes to the operator specific checkers
class QDQChecker : public ConstraintChecker {
 public:
  QDQChecker() = default;

  bool operator()(const Graph& graph, const Node& node) const override {
    std::vector<const Node*> dq_nodes = graph_utils::FindParentsByType(node, DQOpName);
    std::vector<const Node*> q_nodes = graph_utils::FindChildrenByType(node, QOpName);

    return Check(graph, node, dq_nodes, q_nodes);
  }

 private:
  bool virtual Check(const Graph& graph, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes) const = 0;
};

// Single DQ -> node -> Q.
// Zero point and scale are constant scalars and match
class QDQSimpleChecker : public QDQChecker {
 public:
  QDQSimpleChecker();

 private:
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  const ConstantScalarChecker dq_scale_is_constant_scalar_;
  const ConstantScalarChecker dq_zero_point_is_constant_scalar_;
  const ConstantScalarChecker q_scale_is_constant_scalar_;
  const ConstantScalarChecker q_zero_point_is_constant_scalar_;
};

// 1 DQ in, 1 Q out, no graph outputs, constant scalar zp and scale, matching zp and scale
class QDQNodePairChecker : public ConstraintChecker {
 public:
  QDQNodePairChecker(InOutDefSlot dq_slot, InOutDefSlot q_slot);

 private:
  bool operator()(const Graph& graph, const Node& node) const override;

  const InOutDefSlot dq_slot_;
  //const InOutOpTypeChecker dq_in_;
  const ConstantScalarChecker dq_scale_is_constant_scalar_;
  const ConstantScalarChecker dq_zero_point_is_constant_scalar_;
  const InOutDefSlot q_slot_;
  //const InOutOpTypeChecker q_out_;
  const ConstantScalarChecker q_scale_is_constant_scalar_;
  const ConstantScalarChecker q_zero_point_is_constant_scalar_;
};

struct NodeSelectionRules {
  std::unordered_map<std::string, std::vector<int>> ops_and_versions_;

  // need to use unique_ptr as the values are polymorphic.
  std::vector<std::unique_ptr<ConstraintChecker>> rules;
};

//struct ArgHandler {
//  virtual NodeArg* GetNodeArg(Graph& /*graph*/, Node& /*orig_node*/) const = 0;
//  virtual void MoveEdges(Graph& /*graph*/, Node& /*orig_node*/, Node& /*new_node*/) const = 0;
//  virtual ~ArgHandler() = default;
//};
//
//// ArgHandler if a new initializer value was created
//struct CreateInputOutput : public ArgHandler {
//  CreateInputOutput(const std::string& name, ONNX_NAMESPACE::TypeProto type)
//      : name_{name}, type_{type} {}
//
//  NodeArg* GetNodeArg(Graph& graph, Node& /*orig_node*/) const override {
//    // create the new node arg for the initializer and add it to the graph
//    auto new_name = graph.GenerateNodeArgName(name_);
//
//    graph.GetOrCreateNodeArg(new_name, &type_);
//  }
//
//  void MoveEdges(Graph& /*graph*/, Node& /*orig_node*/, Node& /*new_node*/) const override {
//    return;
//  }
//
// private:
//  const std::string name_;
//  const ONNX_NAMESPACE::TypeProto type_;
//};
//
//struct CreateNodeRule {
//  std::string op_type;
//  std::string domain;
//  std::string name;  // generated?
//
//  std::vector<NodeAndArg> inputs;
//  std::vector<NodeAndArg> outputs;
//
//  Node* operator()(Graph&, std::vector<Node*>&);
//};

// actions that are applied to a set of nodes identified during selection
struct Action {
  virtual Status operator()(Graph&, std::vector<Node*>& nodes) = 0;

  // helper to validate the index when fetching a node
  Node* GetNode(size_t index, std::vector<Node*>& nodes, bool required = true) {
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

  Status operator()(Graph& graph, std::vector<Node*>& nodes) override {
    Node& src = *GetNode(source_.node_idx, nodes);
    Node& dest = *GetNode(target_.node_idx, nodes);

    ORT_RETURN_IF_ERROR(MoveNodeArg(src, dest));
    ORT_RETURN_IF_ERROR(MoveEdges(graph, src, dest));
  }

 private:
  Status MoveNodeArg(Node& src, Node& dest) const {
    auto& src_defs = (source_.in_out_slot.in_out == Direction::kInput)
                         ? src.MutableInputDefs()
                         : src.MutableOutputDefs();

    auto& dest_defs = (target_.in_out_slot.in_out == Direction::kInput)
                          ? dest.MutableInputDefs()
                          : dest.MutableOutputDefs();

    if (source_.in_out_slot.idx >= src_defs.size() ||
        target_.in_out_slot.idx >= dest_defs.size()) {
      // TODO improve error message
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Index out of range");
    }

    dest_defs[target_.in_out_slot.idx] = src_defs[source_.in_out_slot.idx];

    return Status::OK();
  }

  Status MoveEdges(Graph& graph, Node& src, Node& dest) const {
    if (source_.in_out_slot.in_out == Direction::kInput) {
      // move input edge if present
      auto iter = std::find_if(src.InputEdgesBegin(), src.InputEdgesEnd(),
                               [this](const Node::EdgeEnd& edge) {
                                 return (edge.GetDstArgIndex() == source_.in_out_slot.idx);
                               });

      // initializer or graph input doesn't have an edge so either zero or one edges to process
      if (iter != src.InputEdgesEnd()) {
        const Node& iter_node = iter->GetNode();
        graph.RemoveEdge(iter_node.Index(), src.Index(), iter->GetSrcArgIndex(), source_.in_out_slot.idx);
        graph.AddEdge(iter_node.Index(), dest.Index(), iter->GetSrcArgIndex(), target_.in_out_slot.idx);
      }

    } else {
      // otherwise we need to move all output edges (if any)
      auto iter = std::find_if(src.OutputEdgesBegin(), src.OutputEdgesEnd(),
                               [this](const Node::EdgeEnd& edge) {
                                 return (edge.GetSrcArgIndex() == source_.in_out_slot.idx);
                               });

      for (const auto end = src.OutputEdgesEnd(); iter != end; ++iter) {
        const Node& iter_node = iter->GetNode();
        graph.RemoveEdge(src.Index(), iter_node.Index(), source_.in_out_slot.idx, iter->GetSrcArgIndex());
        graph.AddEdge(dest.Index(), iter->GetNode().Index(), target_.in_out_slot.idx, iter->GetDstArgIndex());
      }
    }
  }

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
  }

 private:
  const std::vector<size_t> nodes_to_remove_;
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

  std::vector<size_t> nodes_to_remove_;  // index into 'nodes' arg operator() is called with
};

//struct ReplaceWithNew : public Action {
// protected:
//  ReplaceWithNew(std::unique_ptr<CreateNodeRule> creator,
//                 std::initializer_list<size_t> nodes_to_remove,
//                 std::initializer_list<std::unique_ptr<ArgHandler>> input_handlers,
//                 std::initializer_list<std::unique_ptr<ArgHandler>> output_handlers)
//      : node_creation_{std::move(creator)},
//        nodes_to_remove_{nodes_to_remove},
//        input_handlers_{input_handlers},
//        output_handlers_{output_handlers} {
//  }
//
// private:
//  Status operator()(Graph&, std::vector<Node*>& nodes) override {
//    return Status::OK();
//  }
//
//  std::unique_ptr<CreateNodeRule> node_creation_;
//
//  // 2 scenarios.
//  // Drop DQ and D nodes and move edges to existing middle node
//  // Create new QLinear node and move edges from existing DQ, Q and middle nodes depending on the op
//
//  std::vector<size_t> nodes_to_remove_;  // index into 'nodes' arg operator() is called with
//
//  // handlers to move inputs/outputs.
//  // direction can be inferred by whether a node is going to be removed or not
//  //std::vector<std::pair<InOutDefSlot, std::unique_ptr<ArgHandler>>> in_out_handlers_;
//  std::vector<std::unique_ptr<ArgHandler>> input_handlers_;
//  std::vector<std::unique_ptr<ArgHandler>> output_handlers_;
//};

struct Rules {
  NodeSelectionRules NodeSelection;
  std::vector<std::unique_ptr<Action>> Actions;
};

}  // namespace QDQ
}  // namespace onnxruntime
