// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {

SelectorActionTransformer::SelectorActionTransformer(
    const std::string& name,
    std::vector<std::unique_ptr<SelectorAndAction>>&& selectors_and_actions)
    : GraphTransformer(name),
      selectors_and_actions_{std::move(selectors_and_actions)} {
  // setup a map so we lookup by operator type efficiently
  for (const auto& selector_and_actions : selectors_and_actions_) {
    for (const auto& op_info : selector_and_actions->ops_and_versions) {
      bool inserted = op_type_to_selector_and_action_.insert({op_info.first, &*selector_and_actions}).second;

      ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
    }
  }
}

Status SelectorActionTransformer::MatchAndProcess(Graph& graph, Node& node, bool& modified,
                                                  const logging::Logger& logger) const {
  Status status = Status::OK();

  do {
    // TODO: for now this just needs to support ONNX ops. If we ever had a transformer that was going to
    // target non-ONNX ops we'd need to rework a few things to include the op domain in the matches
    if (node.Domain() != kOnnxDomain) {
      break;
    }

    auto op_rule = op_type_to_selector_and_action_.find(node.OpType());
    if (op_rule == op_type_to_selector_and_action_.cend()) {
      break;
    }

    const auto& selector_and_actions = *op_rule->second;

    // check the supported versions if specified
    const auto& versions = selector_and_actions.ops_and_versions.find(node.OpType())->second;
    if (!versions.empty()) {
      if (std::find(versions.cbegin(), versions.cend(), node.SinceVersion()) == versions.cend()) {
        break;
      }
    }

    std::unique_ptr<NodesToOptimize> selections;
    if (!(*selector_and_actions.selector)(graph, node, selections)) {
      break;
    }

    LOGS(logger, VERBOSE) << "Matched " << node.OpType();

    // TODO: To set this up for saving in an ORT format model we would just use a simple Action to save the
    // selected node indexes. May be easiest to store in the Graph instance so we don't need to worry about
    // tracking a graph id across subgraphs

    // TODO: Should we require a single Action at this level to make life easier in terms of recreating the instance
    // that will process the nodes in a minimal build. It can derive from a generic MultiAction Action that provides
    // storage of all the individual actions to run.
    status = (*selector_and_actions.action)(graph, *selections);
    if (!status.IsOK()) {
      break;
    }

    modified = true;
  } while (false);

  return status;
}

Status SelectorActionTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                            const logging::Logger& logger) const {
  // TODO: Is there any reason to create a new GraphViewer? Do we need the different topological sort or can we
  // just use graph.GetNodesInTopologicalOrder() and avoid the overhead of re-sorting.
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto* node = graph.GetNode(index);
    if (node == nullptr) {
      continue;  // was removed by this transformer
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    // TODO: would be more typical to define the supported EP type/s during construction
    // and use GraphTransformer::GetCompatibleExecutionProviders, but doesn't really matter when there's only only
    if (node->GetExecutionProviderType() == kCpuExecutionProvider) {
      ORT_RETURN_IF_ERROR(MatchAndProcess(graph, *node, modified, logger));
    }
  }
  return Status::OK();
}

}  // namespace onnxruntime
