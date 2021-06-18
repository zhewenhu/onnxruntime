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
  virtual Status operator()(Graph&, const NodesToOptimize& selected_nodes) const = 0;
  virtual ~Action() = default;

 protected:
  Action() = default;
};

// helper to assembly multiple actions into a single instance. We do this to keep SelectionActionTransformer simpler
struct MultiAction : public Action {
  MultiAction(std::vector<std::unique_ptr<Action>>&& actions) : actions_{std::move(actions)} {}

  Status operator()(Graph& graph, const NodesToOptimize& selected_nodes) const override {
    for (const auto& action : actions_) {
      ORT_RETURN_IF_ERROR((*action)(graph, selected_nodes));
    }

    return Status::OK();
  }

  // can't copy/assign actions_
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(MultiAction);

 private:
  std::vector<std::unique_ptr<Action>> actions_;
};

// Move a value from one node to another and adjusts edges accordingly
struct MoveInputOutput : public Action {
  MoveInputOutput(const NodeAndMoveInfo& move_info) : move_info_(move_info) {
  }

  Status operator()(Graph& graph, const NodesToOptimize& selected_nodes) const override;

 private:
  const NodeAndMoveInfo move_info_;
};

// Remove selected nodes that the Action applies to based on index.
struct RemoveNodes : public Action {
  RemoveNodes(const NodesToOptimize::NodeIndexes& nodes_to_remove) : nodes_to_remove_{nodes_to_remove} {}

  Status operator()(Graph& graph, const NodesToOptimize& node_group) const override;

 private:
  const NodesToOptimize::NodeIndexes nodes_to_remove_;
};

// Remove all nodes that the Action applies to which no longer produce consumed outputs.
// NOTE: This requires any output edges to have been removed previously.
struct RemoveAllNodes : public Action {
  Status operator()(Graph& graph, const NodesToOptimize& selected_nodes) const override;
};

// Merge multiple nodes into an existing nodes.
// Input/output info in value_moves defines what moves to the target node.
// Edge moves/removal will be automatically handled.
// nodes_to_remove defines the nodes that are no longer needed after the merge.
struct MergeIntoExisting : public Action {
  MergeIntoExisting(const std::initializer_list<NodeAndMoveInfo>& value_moves,
                    const NodesToOptimize::NodeIndexes& nodes_to_remove)
      : node_remover_{nodes_to_remove} {
    for (const auto& value : value_moves) {
      value_movers_.push_back(MoveInputOutput(value));
    }
  }

 private:
  Status operator()(Graph&, const NodesToOptimize& selected_nodes) const override;

  std::vector<MoveInputOutput> value_movers_;
  const RemoveNodes node_remover_;
};

}  // namespace onnxruntime
