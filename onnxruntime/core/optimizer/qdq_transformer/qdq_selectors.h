// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/constraint_checkers.h"
#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {
class Graph;
class Node;

// Base QDQ checker. Provides the DQ and Q nodes to the operator specific checkers
class QDQSelector : public NodeSelector {
 public:
  bool operator()(Graph& graph, const Node& node, std::vector<Node*>& selection) const override;

 protected:
  QDQSelector() = default;

  // override if you need to add entries for missing optional DQ inputs
  // Called post-Check if Check returned `true`
  virtual void AdjustDQNodes(std::vector<const Node*>& /*dq_nodes*/) const {}

  // base check that we have the expected number of QDQ inputs/outputs, and `node` isn't producing a graph output.
  // num_dq_inputs defaults to the number of inputs `node` has if not explicitly specified
  bool CheckQDQNodes(const Graph& graph, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes,
                     size_t num_dq_inputs = std::numeric_limits<size_t>::max()) const;

 private:
  bool virtual Check(const Graph& graph, const Node& node,
                     const std::vector<const Node*>& dq_nodes,
                     const std::vector<const Node*>& q_nodes) const = 0;
};

// Single DQ -> node that does not change data -> Q.
// Zero point and scale are constant scalars and must match
class QDQDropDQDNodesSelector : public QDQSelector {
 public:
  QDQDropDQDNodesSelector();

 private:
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  const ConstantScalarChecker dq_scale_is_constant_scalar_;
  const ConstantScalarChecker dq_zero_point_is_constant_scalar_;
  const ConstantScalarChecker q_scale_is_constant_scalar_;
  const ConstantScalarChecker q_zero_point_is_constant_scalar_;
};

// single input. default is to only support uint8.
class QDQUnarySelector : public QDQSelector {
 public:
  QDQUnarySelector(bool int8_allowed = false) : int8_allowed_{int8_allowed} {}

 private:
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  bool int8_allowed_;
};

// 2 DQ nodes providing input -> node -> Q
class QDQBinarySelector : public QDQSelector {
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;
};

//
class QDQConvSelector : public QDQSelector {
  bool Check(const Graph& graph, const Node& node,
             const std::vector<const Node*>& dq_nodes,
             const std::vector<const Node*>& q_nodes) const override;

  void AdjustDQNodes(std::vector<const Node*>& dq_nodes) const override;
};

}  // namespace onnxruntime
