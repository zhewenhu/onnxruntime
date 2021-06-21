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
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQDropDQDNodesSelector());

  // Move DQ input 0 to target input 0.
  // Move Q output 0 to target output 0.
  std::vector<NodeAndMoveInfo> moves{
      MoveToSlot(dq, ArgType::kInput, 0, ArgType::kInput, 0),
      MoveToSlot(q, ArgType::kOutput, 0, ArgType::kOutput, 0)};

  std::unique_ptr<Action> action(new MergeIntoExisting(std::move(moves)));

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Gather", {}},
                                                                              {"Reshape", {}},
                                                                              {"Transpose", {}},
                                                                              {"MaxPool", {12}}},
                                             std::move(selector),
                                             std::move(action));
}

std::unique_ptr<SelectorAndAction> UnaryOpQDQRules() {
  // 3 nodes. DQ, target, Q
  // Replace with QLinear version of operator. Delete all original nodes.
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQUnarySelector());

  std::vector<std::unique_ptr<Action>> actions;
  ADD_ACTION(actions, QDQ::SetOptionalZeroPoint);  // update the DQ and Q nodes

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq, ArgType::kInput),                            // append all inputs from dq to new node
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),   // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType ::kInput),  // append zp (input 2) from q
      MoveAll(q, ArgType::kOutput)};

  ADD_ACTION(actions, QDQ::ReplaceWithQLinear,  // create new QLinear node to replace target
             kMSDomain,                         // new operator is in MS domain
             std::move(moves));

  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"AveragePool", {}}},
                                             std::move(selector),
                                             std::move(all_actions));
}

std::unique_ptr<SelectorAndAction> BinaryOpQDQRules() {
  // 4 nodes. 2 x DQ for inputs, target, Q
  // Replace with QLinear version of operator. Delete all original nodes.
  NTO::NodeLocation dq1{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq2{NTO::NodeType::kInput, 1};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQBinarySelector());

  std::vector<std::unique_ptr<Action>> actions;

  // set the version point on the two DQ input nodes and the Q output node
  ADD_ACTION(actions, QDQ::SetOptionalZeroPoint);

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq1, ArgType::kInput),                           // append all inputs from dq1 to new node
      MoveAll(dq2, ArgType::kInput),                           // append all inputs from dq2
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),   // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType ::kInput),  // append zp (input 2) from q
      MoveAll(q, ArgType::kOutput)};                           // and use the outputs from q

  ADD_ACTION(actions, QDQ::ReplaceWithQLinear,  // create new QLinear node to replace target
             kMSDomain,                         // new operator is in MS domain
             std::move(moves));

  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Add", {}},
                                                                              {"Mul", {}}},
                                             std::move(selector),
                                             std::move(all_actions));
}

std::unique_ptr<SelectorAndAction> VariadicOpQDQRules() {
  // 0=variadic DQ nodes 2=target, 3=Q
  // Replace with QLinear version of operator. Delete all original nodes.
  NTO::NodeLocation variadic_dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQVariadicSelector());

  std::vector<std::unique_ptr<Action>> actions;

  // set the version point on the DQ input nodes and the Q output node
  ADD_ACTION(actions, QDQ::SetOptionalZeroPoint);

  std::vector<NodeAndMoveInfo> moves{
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),   // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType ::kInput),  // append zp (input 2) from q
      MoveAll(variadic_dq, ArgType::kInput),                   // append all inputs from all dq nodes
      MoveAll(q, ArgType::kOutput)};                           // and use the outputs from q

  ADD_ACTION(actions, QDQ::ReplaceWithQLinear,
             kMSDomain,
             std::move(moves));

  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Concat", {}}},
                                             std::move(selector),
                                             std::move(all_actions));
}

std::unique_ptr<SelectorAndAction> ConvQDQRules() {
  // 4 or 5 Nodes. 0=DQ X, 1=DQ W, 2=DQ B (optional), 3=Conv, 4=Q
  // Handle the DQ input for the Bias being optional.
  // Replace Conv with QLinearConv
  // Delete all original nodes
  NTO::NodeLocation dq_x{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq_w{NTO::NodeType::kInput, 1};
  NTO::NodeLocation dq_bias{NTO::NodeType::kInput, 2};
  // NTO::NodeLocation target{NTO::NodeType::kTarget, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQConvSelector());

  std::vector<std::unique_ptr<Action>> actions;

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq_x, ArgType::kInput),                                     // append all inputs from x
      MoveAll(dq_w, ArgType::kInput),                                     // append all inputs from w
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),              // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType ::kInput),             // append zp (input 2) from q
      MoveAndAppend(dq_bias, ArgType::kInput, 0, ArgType::kInput, true),  // (optional) append bias
      MoveAll(q, ArgType::kOutput)};                                      // and use the outputs from q

  ADD_ACTION(actions, QDQ::ReplaceWithQLinear,
             kOnnxDomain,  // QLinearConv is from ONNX
             std::move(moves));

  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Conv", {}}},
                                             std::move(selector),
                                             std::move(all_actions));
}

std::unique_ptr<SelectorAndAction> MatMulQDQRules() {
  // 3 or 4 nodes. 2 x DQ for inputs, target, optional Q
  // Replace with QLinearMatMul if Q found, or MatMulIntegerToFloat if not.
  // Delete all original nodes.

  std::unique_ptr<NodeSelector> selector(new QDQMatMulSelector());

  std::vector<std::unique_ptr<Action>> actions;

  // set the version point on the two DQ input nodes and the Q output node
  ADD_ACTION(actions, QDQ::SetOptionalZeroPoint);
  ADD_ACTION(actions, QDQ::MatMulAction);
  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"MatMul", {}}},
                                             std::move(selector),
                                             std::move(all_actions));
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
