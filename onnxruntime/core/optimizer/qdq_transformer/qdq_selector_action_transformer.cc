// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/qdq_actions.h"
#include "core/optimizer/qdq_transformer/qdq_selector_action_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_selectors.h"

namespace onnxruntime {

namespace {

using NTO = onnxruntime::NodesToOptimize;

#define ADD_ACTION(container, action, ...) \
  container.push_back(std::unique_ptr<Action>(new action(__VA_ARGS__)))

// create rules for ops that don't change the data
std::unique_ptr<SelectorAndAction> DropQDQNodesRules() {
  // 3 nodes. DQ, target, Q. Merge into target and remove DQ and Q.

  std::unique_ptr<NodeSelector> selector(new QDQ::DropDQDNodesSelector());
  std::unique_ptr<Action> action(new MergeIntoTarget());

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Gather", {}},
                                                                              {"Reshape", {}},
                                                                              {"Transpose", {}},
                                                                              {"MaxPool", {12}}},
                                             std::move(selector),
                                             std::move(action));
}

std::unique_ptr<SelectorAndAction> UnaryOpQDQRules() {
  // 3 nodes. DQ, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.
  std::unique_ptr<NodeSelector> selector(new QDQ::UnarySelector());

  std::unique_ptr<Action> action(new QDQ::UnaryReplaceWithQLinear(kMSDomain));

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"AveragePool", {}}},
                                             std::move(selector),
                                             std::move(action));
}

std::unique_ptr<SelectorAndAction> BinaryOpQDQRules() {
  // 4 nodes. 2 x DQ for inputs, target, Q
  // Replace with internal QLinear version of operator. Delete all original nodes.

  std::unique_ptr<NodeSelector> selector(new QDQ::BinarySelector());
  std::unique_ptr<Action> action(new QDQ::BinaryReplaceWithQLinear(kMSDomain));
  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Add", {}},
                                                                              {"Mul", {}}},
                                             std::move(selector),
                                             std::move(action));
}

std::unique_ptr<SelectorAndAction> VariadicOpQDQRules() {
  // 0=variadic DQ nodes 2=target, 3=Q
  // Replace with QLinear version of operator. Delete all original nodes.
  NTO::NodeLocation variadic_dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQ::VariadicSelector());
  std::unique_ptr<Action> action(new QDQ::VariadicReplaceWithQLinear(kMSDomain));
  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Concat", {}}},
                                             std::move(selector),
                                             std::move(action));
}

std::unique_ptr<SelectorAndAction> ConvQDQRules() {
  // 4 or 5 Nodes. 0=DQ X, 1=DQ W, 2=DQ B (optional), 3=Conv, 4=Q
  // Handle the DQ input for the Bias being optional.
  // Replace Conv with QLinearConv
  // Delete all original nodes
  NTO::NodeLocation dq_x{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq_w{NTO::NodeType::kInput, 1};
  NTO::NodeLocation dq_bias{NTO::NodeType::kInput, 2};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQ::ConvSelector());
  std::unique_ptr<Action> action(new QDQ::ConvReplaceWithQLinear());
  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Conv", {}}},
                                             std::move(selector),
                                             std::move(action));
}

std::unique_ptr<SelectorAndAction> MatMulQDQRules() {
  // 3 or 4 nodes. 2 x DQ for inputs, target, optional Q
  // Replace with QLinearMatMul if Q found, or MatMulIntegerToFloat if not.
  // Delete all original nodes.

  std::unique_ptr<NodeSelector> selector(new QDQ::MatMulSelector());
  std::unique_ptr<Action> action(new QDQ::MatMulReplaceWithQLinear());
  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"MatMul", {}}},
                                             std::move(selector),
                                             std::move(action));
}

static std::vector<std::unique_ptr<SelectorAndAction>> CreateQDQSelectorActionEntries() {
  std::vector<std::unique_ptr<SelectorAndAction>> qdq_selector_action_entries;

  // can't use an initializer list with unique_ptr values as that involves a copy,
  // so have to push_back each entry individually
  qdq_selector_action_entries.reserve(8);
  qdq_selector_action_entries.push_back(std::move(DropQDQNodesRules()));
  qdq_selector_action_entries.push_back(std::move(UnaryOpQDQRules()));
  qdq_selector_action_entries.push_back(std::move(BinaryOpQDQRules()));
  qdq_selector_action_entries.push_back(std::move(VariadicOpQDQRules()));
  qdq_selector_action_entries.push_back(std::move(ConvQDQRules()));
  qdq_selector_action_entries.push_back(std::move(MatMulQDQRules()));

  return qdq_selector_action_entries;
}

}  // namespace

QDQSelectorActionTransformer::QDQSelectorActionTransformer()
    : SelectorActionTransformer{
          "QDQSelectorActionTransformer",
          CreateQDQSelectorActionEntries()} {
}

}  // namespace onnxruntime
