// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/graph/graph_utils.h"  // TODO: Minimize usage of this given we want to use Actions in a minimal build
#include "core/optimizer/selectors_actions/helpers.h"

namespace onnxruntime {

class Graph;
class Node;

// actions that are applied to a set of nodes identified during selection
struct Action {
  virtual Status Run(Graph&, const NodesToOptimize& selected_nodes) const = 0;
  virtual ~Action() = default;

 protected:
  Action() = default;
};

// helper to assembly multiple actions into a single instance. We do this to keep SelectionActionTransformer simpler
struct MultiAction : public Action {
  MultiAction(std::vector<std::unique_ptr<Action>>&& actions) : actions_{std::move(actions)} {}

  Status Run(Graph& graph, const NodesToOptimize& selected_nodes) const override {
    for (const auto& action : actions_) {
      ORT_RETURN_IF_ERROR(action->Run(graph, selected_nodes));
    }

    return Status::OK();
  }

  // can't copy/assign actions_
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(MultiAction);

 private:
  std::vector<std::unique_ptr<Action>> actions_;
};

// Safely remove nodes that the Action applies to which no longer produce consumed outputs.
// Output edges to nodes in selected_nodes are ignored when determining if it's safe to remove a node.
// Set `preserve_target_node` for the NodesToOptimize::Target node to not be removed.
struct RemoveNodes : public Action {
  RemoveNodes(bool preserve_target_node = false) : preserve_target_node_{preserve_target_node} {
  }

  Status Run(Graph& graph, const NodesToOptimize& selected_nodes) const override;

 private:
  bool preserve_target_node_;
};

// Merge one input and/or one output node into the target node.
//   - inputs from the input node, if present, will become the inputs of the target node
//   - outputs from the output node, if present, will become the outputs of the target node
// The input and/or output node will be removed after the merge. The target node will not.
struct MergeIntoTarget : public Action {
 private:
  Status Run(Graph&, const NodesToOptimize& selected_nodes) const override;

  RemoveNodes node_remover_{true};  // preserve target node when removing selected_nodes
};

struct ReplaceWithNew : public Action {
  // provide NodeLocation for source node, and ValueMoveInfo for the value to move to the replacement node
  ReplaceWithNew(const std::string& domain,
                 const std::string& op_name,
                 std::vector<NodeAndMoveInfo>&& value_moves);

  Status Run(Graph&, const NodesToOptimize& selected_nodes) const override;

 private:
  // support usage where operator name is determined at runtime from the selected nodes
  virtual std::string OpType(const NodesToOptimize&) const { return op_; }

  // TODO: setup mechanism to create a new NodeArg
  // If we use resize on the input defs we can do the moves and directly populate the slot with a new NodeArg
  // but this may not be needed for QDQ.

  const std::string domain_;
  const std::string op_;
  std::vector<NodeAndMoveInfo> value_moves_;
  RemoveNodes node_remover_;
};

}  // namespace onnxruntime
