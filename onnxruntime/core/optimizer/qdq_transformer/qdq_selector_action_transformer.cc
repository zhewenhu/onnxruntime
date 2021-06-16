// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/qdq_actions.h"
#include "core/optimizer/qdq_transformer/qdq_selector_action_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_selectors.h"

namespace onnxruntime {
namespace {

// Helpers to make the 'move' configuration more easily read
//
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

#define ADD_ACTION(container, action, ...) \
  container.push_back(std::unique_ptr<Action>(new action(__VA_ARGS__)))

// create rules for ops that don't change the data
std::unique_ptr<SelectorAndAction> SimpleQDQRules() {
  // 3 nodes. 0=DQ, 1=target, 2=Q. Merge into target and remove DQ and Q.
  // Move DQ input 0 to target input 0.
  // Move Q output 0 to other output 0.
  // Delete DQ and Q
  std::unique_ptr<NodeSelector> selector(new QDQSimpleSelector());

  std::vector<std::unique_ptr<Action>> actions;
  ADD_ACTION(actions, MergeIntoExisting,
             {MoveInput(0, 0, 1, 0),    // input 0 from DQ moves to target
              MoveOutput(2, 0, 1, 0)},  // output 0 from Q moves to target
             {0, 2});

  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};

  return std::make_unique<SelectorAndAction>(SelectorAndAction{{{"Gather", {}},
                                                                {"Reshape", {}},
                                                                {"Transpose", {}},
                                                                {"MaxPool", {12}}},
                                                               std::move(selector),
                                                               std::move(all_actions)});
}

std::unique_ptr<SelectorAndAction> BinaryOpQDQRules() {
  // 4 nodes. 0=DQ for inputA, 1=DQ for inputB, 2=target, 3=Q
  // Replace with QLinear version of operator. Delete all original nodes.
  std::unique_ptr<NodeSelector> selector(new QDQBinarySelector());

  std::vector<std::unique_ptr<Action>> actions;
  ADD_ACTION(actions, QDQ::SetOptionalZeroPoint, {0, 1, 3});  // update the DQ and Q nodes
  ADD_ACTION(actions, QDQ::ReplaceWithQLinear,
             2,                       // replace node 2
             kMSDomain,               // QLinearAdd and QLinearMul are internal ops
             {MoveInput(0, -1, -1),   // append all inputs from dq[0]
              MoveInput(1, -1, -1),   // append all inputs from dq[1]
              MoveInput(3, 1, -1),    // append scale from q[0]
              MoveInput(3, 2, -1),    // append zp from q[0]
              MoveOutput(3, -1, -1)}  // and use the outputs from q[0]
  );

  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};
  return std::make_unique<SelectorAndAction>(SelectorAndAction{{{"Add", {}},
                                                                {"Mul", {}}},
                                                               std::move(selector),
                                                               std::move(all_actions)});
}

std::unique_ptr<SelectorAndAction> ConvQDQRules() {
  // 4 or 5 Nodes. 0=DQ X, 1=DQ W, 2=DQ B (optional), 3=Conv, 4=Q
  // Handle the DQ input for the Bias being optional.
  // Replace Conv with QLinearConv
  // Delete all original nodes
  std::unique_ptr<NodeSelector> selector(new QDQConvSelector());

  std::vector<std::unique_ptr<Action>> actions;
  ADD_ACTION(actions, QDQ::ReplaceWithQLinear,
             3,                       // replace node 3
             kOnnxDomain,             // QLinearConv is from ONNX
             {MoveInput(0, -1, -1),   // append all inputs from dq[0]
              MoveInput(1, -1, -1),   // append all inputs from dq[1]
              MoveInput(4, 1, -1),    // append scale from q[0]
              MoveInput(4, 2, -1),    // append zp from q[0]
              MoveInput(2, 0, -1),    // (optional) append input dq[2][0]
              MoveOutput(4, -1, -1)}  // and use the outputs from q[0]
  );

  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};

  return std::make_unique<SelectorAndAction>(SelectorAndAction{{{"Conv", {}}},
                                                               std::move(selector),
                                                               std::move(all_actions)});
}

static std::vector<std::unique_ptr<SelectorAndAction>>&& CreateQDQSelectorActionEntries() {
  std::vector<std::unique_ptr<SelectorAndAction>> qdq_selector_action_entries;

  // can't use an initializer list with unique_ptr values so have to push_back each entry individually
  qdq_selector_action_entries.reserve(8);
  qdq_selector_action_entries.push_back(SimpleQDQRules());
  qdq_selector_action_entries.push_back(BinaryOpQDQRules());
  qdq_selector_action_entries.push_back(ConvQDQRules());

  return std::move(qdq_selector_action_entries);
}

}  // namespace

QDQSelectorActionTransformer::QDQSelectorActionTransformer()
    : SelectorActionTransformer{
          "QDQSelectorActionTransformer",
          std::move(CreateQDQSelectorActionEntries())} {
}

}  // namespace onnxruntime
