// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/actions.h"

namespace onnxruntime {

class Graph;
class Node;

namespace QDQ {

struct SetOptionalZeroPoint : public Action {
  Status operator()(Graph&, const NodesToOptimize& selected_nodes) const override;

 private:
  static const ONNX_NAMESPACE::TensorProto optional_zero_point_int8_;
  static const ONNX_NAMESPACE::TensorProto optional_zero_point_uint8_;
};

// replace node with QLinear version
// TODO: If we extend this setup to other optimizers this could be lifted into a more generic 'replace with new node'
// implementation. Current version has some QDQ specific logic embedded in it.
struct ReplaceWithQLinear : public ReplaceWithNew {
  // provide NodeLocation for source node, and ValueMoveInfo for the value to move to the replacement node
  ReplaceWithQLinear(const std::string& domain,
                     std::vector<NodeAndMoveInfo>&& value_moves)
      : ReplaceWithNew{domain, "unknown", std::move(value_moves)} {}

 private:
  std::string ReplaceWithQLinear::OpType(const NodesToOptimize& selected_nodes) const override {
    return "QLinear" + selected_nodes.Target().OpType();
  }
};

struct MatMulAction : public Action {
  MatMulAction();

  Status operator()(Graph&, const NodesToOptimize& selected_nodes) const override;

 private:
  ReplaceWithNew matmul_int_to_float_replacer_;
  ReplaceWithQLinear qlinear_matmul_replacer_;
};

}  // namespace QDQ
}  // namespace onnxruntime
