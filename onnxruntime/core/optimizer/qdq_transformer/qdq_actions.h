// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/actions.h"

namespace onnxruntime {

class Graph;
class Node;

namespace QDQ {

struct SetOptionalZeroPoint : public Action {
  SetOptionalZeroPoint(std::initializer_list<NodesToOptimize::NodeLocation> nodes_to_update)
      : nodes_to_update_{nodes_to_update} {}

 private:
  Status operator()(Graph&, const NodesToOptimize& selected_nodes) const override;

  std::vector<NodesToOptimize::NodeLocation> nodes_to_update_;

  static const ONNX_NAMESPACE::TensorProto optional_zero_point_int8_;
  static const ONNX_NAMESPACE::TensorProto optional_zero_point_uint8_;
};

// replace node with QLinear version
// TODO: If we extend this setup to other optimizers this could be lifted into a more generic 'replace with new node'
// implementation. Current version has some QDQ specific logic embedded in it.
struct ReplaceWithQLinear : public Action {
  // provide NodeLocation for source node, and ValueMoveInfo for the value to move to the replacement node
  ReplaceWithQLinear(const std::string& domain,
                     std::initializer_list<NodeAndMoveInfo> value_moves);

 private:
  Status operator()(Graph&, const NodesToOptimize& selected_nodes) const override;

  // TODO: setup mechanism to create a new NodeArg
  // If we use resize on the input defs we can do the moves and directly populate the slot with a new NodeArg
  // but this may not be needed for QDQ.

  const std::string domain_;
  std::vector<NodeAndMoveInfo> value_moves_;
};

}  // namespace QDQ
}  // namespace onnxruntime
