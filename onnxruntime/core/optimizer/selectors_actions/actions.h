// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"  // TODO: Minimize usage of this given we want to use Actions in a minimal build
#include "core/optimizer/selectors_actions/helpers.h"

namespace onnxruntime {

class Graph;
class Node;

// actions that are applied to a set of nodes identified during selection
struct Action {
  virtual Status operator()(Graph&, std::vector<Node*>& nodes) = 0;
  virtual ~Action() = default;

  // helper to validate the index when fetching a node
  Node* GetNode(size_t index, const std::vector<Node*>& nodes, bool required = true) {
    Node* node = nullptr;
    ORT_ENFORCE(index < nodes.size() && ((node = nodes[index]) != nullptr || !required));

    return node;
  }

 protected:
  Action() = default;
};

// helper to assembly multiple actions into a single instance. We do this to keep SelectionActionTransformer simpler
struct MultiAction : public Action {
  MultiAction(std::vector<std::unique_ptr<Action>>&& actions) : actions_{std::move(actions)} {}

  Status operator()(Graph& graph, std::vector<Node*>& nodes) override {
    for (const auto& action : actions_) {
      ORT_RETURN_IF_ERROR((*action)(graph, nodes));
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
  MoveInputOutput(NodeAndArg src, NodeAndArg dest) : source_{src}, target_{dest} {
  }

  Status operator()(Graph& graph, std::vector<Node*>& nodes) override;

 private:
  NodeAndArg source_;
  NodeAndArg target_;
};

// Remove selected nodes that the Action applies to based on index
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

// Remove all nodes that the Action applies to
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

// Merge multiple nodes into an existing nodes.
// Input/output info in value_moves defines what moves to the target node.
// Edge moves/removal will be automatically handled.
// nodes_to_remove defines the nodes that are no longer needed after the merge.
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

}  // namespace onnxruntime
