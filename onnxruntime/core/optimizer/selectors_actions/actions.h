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

//// Remove selected nodes that the Action applies to based on index.
//struct RemoveNodes : public Action {
//  RemoveNodes(const NodesToOptimize::NodeIndexes& nodes_to_remove)
//      : nodes_to_remove_{nodes_to_remove} {}
//
//  Status operator()(Graph& graph, const NodesToOptimize& node_group) const override;
//
// private:
//  NodesToOptimize::NodeIndexes nodes_to_remove_;
//};

// Remove all nodes that the Action applies to which no longer produce consumed outputs.
// NOTE: This requires any output edges to have been removed previously.
struct RemoveNodes : public Action {
  RemoveNodes(bool preserve_target_node = false) : preserve_target_node_{preserve_target_node} {
  }

  Status operator()(Graph& graph, const NodesToOptimize& selected_nodes) const override;

 private:
  bool preserve_target_node_;
};

// Merge multiple nodes into an existing nodes.
// Input/output info in value_moves defines what moves to the target node.
// Edge moves/removal will be automatically handled.
// nodes_to_remove defines the nodes that are no longer needed after the merge.
struct MergeIntoExisting : public Action {
  MergeIntoExisting(const std::vector<NodeAndMoveInfo>& value_moves) {
    for (const auto& value : value_moves) {
      value_movers_.push_back(MoveInputOutput(value));
    }
  }

 private:
  Status operator()(Graph&, const NodesToOptimize& selected_nodes) const override;

  std::vector<MoveInputOutput> value_movers_;
  RemoveNodes node_remover_{true};  // preserve target node
};

struct ReplaceWithNew : public Action {
  // provide NodeLocation for source node, and ValueMoveInfo for the value to move to the replacement node
  ReplaceWithNew(const std::string& domain,
                 const std::string& op_name,
                 std::vector<NodeAndMoveInfo>&& value_moves);

  Status operator()(Graph&, const NodesToOptimize& selected_nodes) const override;

 private:
  // support usage where operator name is determined at runtime from the selected nodes
  virtual std::string OpType(const NodesToOptimize&) const { return op_; }

  // TODO: setup mechanism to create a new NodeArg
  // If we use resize on the input defs we can do the moves and directly populate the slot with a new NodeArg
  // but this may not be needed for QDQ.

  const std::string domain_;
  const std::string op_;
  std::vector<NodeAndMoveInfo> value_moves_;
};

}  // namespace onnxruntime
