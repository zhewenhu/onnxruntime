// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_rules_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/transformer_rules.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
namespace QDQ {
class RulesTransformerImpl {
 public:
  RulesTransformerImpl(Graph& graph, const std::vector<std::unique_ptr<Rules>>& rules)
      : graph_{graph} /*, rules_{rules} */ {
    for (const auto& rule : rules) {
      for (const auto& op_info : rule->NodeSelection.ops_and_versions_) {
        bool inserted = op_to_rules_.insert({op_info.first, &*rule}).second;
        ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
      }
    }
  }

  bool IsMatch(const Node& node) const {
    if (node.Domain() != kOnnxDomain) {
      return false;
    }

    auto op_rule = op_to_rules_.find(node.OpType());
    if (op_rule == op_to_rules_.cend()) {
      return false;
    }

    const auto& rules = *op_rule->second;

    // check the supported versions if specified
    const auto& versions = rules.NodeSelection.ops_and_versions_.find(node.OpType())->second;
    if (!versions.empty()) {
      if (std::find(versions.cbegin(), versions.cend(), node.SinceVersion()) == versions.cend()) {
        return false;
      }
    }

    // use 'for' loop for debugging. switch to std::all_of before checkin
    for (const auto& rule : rules.NodeSelection.rules) {
      if (!(*rule)(graph_, node)) {
        return false;
      }
    }

    bool redo = std::all_of(rules.NodeSelection.rules.cbegin(),
                            rules.NodeSelection.rules.cend(),
                            [this, &node](const std::unique_ptr<ConstraintChecker>& check) {
                              return (*check)(graph_, node);
                            });

    assert(redo);  // all_of should return the same result

    return true;
  }

  Status Process(const Node& /*node*/) const {
    // create new node/s

    // move edges for inputs

    // move edges for outputs

    // remove nodes
    return Status::OK();
  }

 private:
  Graph& graph_;
  //const std::vector<Rules>& rules_;
  //std::unordered_set<std::string> ops_;
  std::unordered_map<std::string, const Rules*> op_to_rules_;
};

#define ADD_RULE(container, rule, ...) \
  container.push_back(std::unique_ptr<ConstraintChecker>(new rule(__VA_ARGS__)))
#define ADD_ACTION(container, action, ...) \
  container.push_back(std::unique_ptr<Action>(new action(__VA_ARGS__)))

namespace {

std::pair<NodeAndArg, NodeAndArg> MoveInput(size_t src, size_t src_idx, size_t dest, size_t dest_idx) {
  return {NodeAndArg{src, InOutDefSlot{Direction::kInput, src_idx}},
          NodeAndArg{dest, InOutDefSlot{Direction::kInput, dest_idx}}};
}

std::pair<NodeAndArg, NodeAndArg> MoveOutput(size_t src, size_t src_idx, size_t dest, size_t dest_idx) {
  return {NodeAndArg{src, InOutDefSlot{Direction::kOutput, src_idx}},
          NodeAndArg{dest, InOutDefSlot{Direction::kOutput, dest_idx}}};
}

// create rules for ops that don't change the data
std::unique_ptr<Rules> SimpleQDQRules() {
  std::vector<std::unique_ptr<ConstraintChecker>> constraints;

  // check that input 0 and output 0 are connected to a DQ and Q node,
  ADD_RULE(constraints, QDQNodePairChecker, InOutDefSlot{Direction::kInput, 0}, InOutDefSlot{Direction::kOutput, 0});

  std::vector<std::unique_ptr<Action>> actions;

  // 3 nodes. 0=DQ, 1=other, 2=Q. Merge into other. Move DQ input 0 to other input 0 and Q output 0 to other output 0.
  // delete nodes 0 and 2
  ADD_ACTION(actions, MergeIntoExisting, {MoveInput(0, 0, 1, 0), MoveOutput(2, 0, 1, 0)}, {0, 2});

  // rules apply to all versions of the operators so empty list of supported versions
  //NodeSelectionRules selection{
  //    {{"Gather", {}},
  //     {"Reshape", {}},
  //     {"Transpose", {}}},
  //    std::move(constraints)};

  return std::make_unique<Rules>(Rules{NodeSelectionRules{
                                           {{"Gather", {}},
                                            {"Reshape", {}},
                                            {"Transpose", {}}},
                                           std::move(constraints)},
                                       std::move(actions)});
}

// create rules for ops with 2 inputs
//Rules BinaryQDQRules() {
//}
}  // namespace

RulesTransformer::RulesTransformer() noexcept
    : GraphTransformer("QDQ::RulesTransformer") {
  rules_.reserve(8);

  rules_.push_back(std::move(SimpleQDQRules()));

  // setup lookup map for all the rules
  for (const auto& rule : rules_) {
    for (const auto& op_info : rule->NodeSelection.ops_and_versions_) {
      const auto& op_name = op_info.first;
      op_to_rules_.insert({op_name, &*rule});
    }
  }
}

Status RulesTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                   const logging::Logger& logger) const {
  RulesTransformerImpl impl(graph, rules_);
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);

    // no point replacing a node with a quantized version if it is producing a graph output
    if (!graph.GetNodeOutputsInGraphOutputs(node).empty()) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
      if (impl.IsMatch(node)) {
        ORT_RETURN_IF_ERROR(impl.Process(node));
      }
    }
  }
  return Status::OK();
}

}  // namespace QDQ
}  // namespace onnxruntime
