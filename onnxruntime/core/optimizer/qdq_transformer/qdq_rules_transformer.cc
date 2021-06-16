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
  RulesTransformerImpl(Graph& graph, const std::vector<std::unique_ptr<SelectorAndActions>>& rules_and_actions)
      : graph_{graph} /*, rules_{rules} */ {
    for (const auto& rule_and_actions : rules_and_actions) {
      for (const auto& op_info : rule_and_actions->ops_and_versions) {
        bool inserted = op_to_rules_and_actions_.insert({op_info.first, &*rule_and_actions}).second;
        ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
      }
    }
  }

  Status MatchAndProcess(const Node& node) const {
    Status status = Status::OK();

    do {
      if (node.Domain() != kOnnxDomain) {
        break;
      }

      auto op_rule = op_to_rules_and_actions_.find(node.OpType());
      if (op_rule == op_to_rules_and_actions_.cend()) {
        break;
      }

      const auto& rules_and_actions = *op_rule->second;

      // check the supported versions if specified
      const auto& versions = rules_and_actions.ops_and_versions.find(node.OpType())->second;
      if (!versions.empty()) {
        if (std::find(versions.cbegin(), versions.cend(), node.SinceVersion()) == versions.cend()) {
          break;
        }
      }

      std::vector<Node*> selections;
      if (!(*rules_and_actions.selector)(graph_, node, selections)) {
        break;
      }

      std::cout << "Matched " << node.OpType() << "\n";

      // NOTE: To replay the modifications we just need to save the Action name and the set of node indexes to pass
      // to it.
      for (const auto& action : rules_and_actions.actions) {
        status = (*action)(graph_, selections);
        if (!status.IsOK()) {
          break;
        }
      }
    } while (false);

    return status;
  }

 private:
  Graph& graph_;
  std::unordered_map<std::string, const SelectorAndActions*> op_to_rules_and_actions_;
};

//#define ADD_RULE(container, rule, ...) \
//  container.push_back(std::unique_ptr<ConstraintChecker>(new rule(__VA_ARGS__)))
#define ADD_ACTION(container, action, ...) \
  container.push_back(std::unique_ptr<Action>(new action(__VA_ARGS__)))

namespace {

// moves between two existing nodes where the dest node idx is known
std::pair<NodeAndArg, NodeAndArg> MoveInput(size_t src_node_idx, int src_arg_idx,
                                            size_t dest_node_idx, int dest_arg_idx) {
  return {NodeAndArg{src_node_idx, InOutDefSlot{Direction::kInput, src_arg_idx}},
          NodeAndArg{dest_node_idx, InOutDefSlot{Direction::kInput, dest_arg_idx}}};
}

std::pair<NodeAndArg, NodeAndArg> MoveOutput(size_t src_node_idx, int src_arg_idx,
                                             size_t dest_node_idx, int dest_arg_idx) {
  return {NodeAndArg{src_node_idx, InOutDefSlot{Direction::kOutput, src_arg_idx}},
          NodeAndArg{dest_node_idx, InOutDefSlot{Direction::kOutput, dest_arg_idx}}};
}

// moves to new node
std::pair<NodeAndArg, InOutDefSlot> MoveInput(size_t src_node_idx, int src_arg_idx, int dest_arg_idx) {
  return {NodeAndArg{src_node_idx, InOutDefSlot{Direction::kInput, src_arg_idx}},
          InOutDefSlot{Direction::kInput, dest_arg_idx}};
}

std::pair<NodeAndArg, InOutDefSlot> MoveOutput(size_t src_node_idx, int src_arg_idx, int dest_arg_idx) {
  return {NodeAndArg{src_node_idx, InOutDefSlot{Direction::kOutput, src_arg_idx}},
          InOutDefSlot{Direction::kOutput, dest_arg_idx}};
}

// create rules for ops that don't change the data
std::unique_ptr<SelectorAndActions> SimpleQDQRules() {
  //std::vector<std::unique_ptr<ConstraintChecker>> constraints;

  // check that input 0 and output 0 are connected to a DQ and Q node,
  //  ADD_RULE(constraints, QDQSimpleSelector);

  std::unique_ptr<NodeSelector> selector(new QDQSimpleSelector());
  std::vector<std::unique_ptr<Action>> actions;

  // 3 nodes. 0=DQ, 1=target, 2=Q. Merge into target and remove DQ and Q.
  // Move DQ input 0 to target input 0.
  // Move Q output 0 to other output 0.
  // Delete DQ and Q
  ADD_ACTION(actions, MergeIntoExisting, {MoveInput(0, 0, 1, 0), MoveOutput(2, 0, 1, 0)}, {0, 2});

  return std::make_unique<SelectorAndActions>(SelectorAndActions{{{"Gather", {}},
                                                                  {"Reshape", {}},
                                                                  {"Transpose", {}},
                                                                  {"MaxPool", {12}}},
                                                                 std::move(selector),
                                                                 std::move(actions)});
}

std::unique_ptr<SelectorAndActions> BinaryOpQDQRules() {
  std::unique_ptr<NodeSelector> selector(new QDQBinarySelector());
  std::vector<std::unique_ptr<Action>> actions;

  // 0=DQ for inputA, 1=DQ for inputB, 2=target, 3=Q

  // set any missing zero point to 0.
  ADD_ACTION(actions, SetOptionalZeroPoint, {0, 1, 3});
  ADD_ACTION(actions, ReplaceWithNew, 2,  // replace node 2
             {MoveInput(0, -1, -1),       // append all inputs from dq[0]
              MoveInput(1, -1, -1),       // append all inputs from dq[1]
              MoveInput(3, 1, -1),        // append scale from q[0]
              MoveInput(3, 2, -1),        // append zp from q[0]
              MoveOutput(3, -1, -1)}      // and use the outputs from q[0]
  );

  return std::make_unique<SelectorAndActions>(SelectorAndActions{{{"Add", {}},
                                                                  {"Mul", {}}},
                                                                 std::move(selector),
                                                                 std::move(actions)});
}
}  // namespace

RulesTransformer::RulesTransformer() noexcept
    : GraphTransformer("QDQ::RulesTransformer") {
  rules_and_actions_.reserve(8);

  rules_and_actions_.push_back(std::move(SimpleQDQRules()));
  rules_and_actions_.push_back(std::move(BinaryOpQDQRules()));

  // setup lookup map for all the rules
  for (const auto& entry : rules_and_actions_) {
    for (const auto& op_info : entry->ops_and_versions) {
      const auto& op_name = op_info.first;
      op_to_rules_and_actions_.insert({op_name, &*entry});
    }
  }
}

Status RulesTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                   const logging::Logger& logger) const {
  RulesTransformerImpl impl(graph, rules_and_actions_);
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto* node = graph.GetNode(index);
    if (node == nullptr) {
      continue;  // was removed by this transformer
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (node->GetExecutionProviderType() == kCpuExecutionProvider) {
      ORT_RETURN_IF_ERROR(impl.MatchAndProcess(*node));
    }
  }
  return Status::OK();
}

}  // namespace QDQ
}  // namespace onnxruntime
