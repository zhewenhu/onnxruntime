// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/selectors_actions/actions.h"

namespace onnxruntime {

class Graph;
class Node;

// Base class for a selector which checks for a match and returns the set of nodes involved.
struct NodeSelector {
  // Select one or more nodes for an Action to process if the constraints are satisfied.
  // `selection` is be ignored if this returns false
  virtual bool operator()(Graph& graph, const Node& node, std::unique_ptr<NodesToOptimize>& selection) const = 0;
  virtual ~NodeSelector() = default;

 protected:
  NodeSelector() = default;
};

struct SelectorAndAction {
  using OpVersionsMap = std::unordered_map<std::string, std::vector<ONNX_NAMESPACE::OperatorSetVersion>>;

  // ctor so we can use make_unique to construct this class
  SelectorAndAction(const OpVersionsMap& ops_and_versions_in,
                    std::unique_ptr<NodeSelector> selector_in,
                    std::unique_ptr<Action> action_in)
      : ops_and_versions{ops_and_versions_in},
        selector{std::move(selector_in)},
        action{std::move(action_in)} {}

  // Operator and supported versions for the node that selection will start from.
  OpVersionsMap ops_and_versions;
  std::unique_ptr<NodeSelector> selector;
  std::unique_ptr<Action> action;

  // can't copy/assign our unique_ptr members
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SelectorAndAction);
};

/**
@Class SelectorActionTransformer
*/
class SelectorActionTransformer : public GraphTransformer {
 protected:
  // TODO: We could make it 1:1 between transformer and SelectorAndAction instead, but that can also be
  // achieved by passing a single entry in selector_and_actions, so this setup is more flexible.
  SelectorActionTransformer(const std::string& name,
                            std::vector<std::unique_ptr<SelectorAndAction>>&& selectors_and_actions);

  // can't copy/assign selectors_and_actions_
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SelectorActionTransformer);

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  Status MatchAndProcess(Graph& graph, Node& node, bool& modified, const logging::Logger& logger) const;

  std::vector<std::unique_ptr<SelectorAndAction>> selectors_and_actions_;
  std::unordered_map<std::string, const SelectorAndAction*> op_type_to_selector_and_action_;
};

}  // namespace onnxruntime
