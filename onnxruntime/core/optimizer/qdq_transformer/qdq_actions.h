// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/actions.h"

namespace onnxruntime {

class Graph;
class Node;

namespace QDQ {

struct SetOptionalZeroPoint : public Action {
  SetOptionalZeroPoint(std::initializer_list<size_t> nodes_to_update)
      : nodes_to_update_{nodes_to_update} {}

 private:
  Status operator()(Graph&, std::vector<Node*>& nodes) override;

  std::vector<size_t> nodes_to_update_;  // index into 'nodes' vector that operator() is called with

  static const ONNX_NAMESPACE::TensorProto optional_zero_point_int8_;
  static const ONNX_NAMESPACE::TensorProto optional_zero_point_uint8_;
};

// replace node with QLinear version
// TODO: If we extend this setup to other optimizers this could be lifted into a more generic 'replace with new node'
// implementation. Current version has some QDQ specific logic embedded in it.
struct ReplaceWithQLinear : public Action {
  ReplaceWithQLinear(size_t node_to_replace,
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

}  // namespace QDQ
}  // namespace onnxruntime
