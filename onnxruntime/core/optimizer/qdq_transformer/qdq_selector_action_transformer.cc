// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/qdq_actions.h"
#include "core/optimizer/qdq_transformer/qdq_selector_action_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_selectors.h"

namespace onnxruntime {

using NTO = onnxruntime::NodesToOptimize;

namespace {

//
// Helpers to make the 'move' configuration more easily read
//

// move specific input/output to slot on target node
NodeAndMoveInfo MoveToSlot(const NTO::NodeLocation& src_node,
                           Direction src_direction, int src_slot,
                           Direction dest_direction, int dest_slot) {
  return NodeAndMoveInfo{src_node,
                         ValueMoveInfo{
                             InOutDefSlot{src_direction, src_slot},      // move from this slot
                             InOutDefSlot{dest_direction, dest_slot}}};  // to this one
}

// move specific input/output and append to target node
// is the source input/output is variadic (i.e. multiple values may need to be moved) set `variadic` to true
NodeAndMoveInfo MoveAndAppend(const NTO::NodeLocation& src_node,
                              Direction src_direction, int src_slot,
                              Direction dest_direction,
                              bool variadic = false) {
  return NodeAndMoveInfo{src_node, ValueMoveInfo{InOutDefSlot{src_direction, src_slot},  // move from this slot
                                                 dest_direction,                         // append here
                                                 variadic}};
}

// move all inputs/outputs from the source node to the target node
// if the last source input is variadic set `variadic` to true
NodeAndMoveInfo MoveAll(const NTO::NodeLocation& src_node,
                        Direction direction,  // moving inputs or outputs
                        bool variadic = false) {
  return NodeAndMoveInfo{src_node, ValueMoveInfo{direction, direction, variadic}};
}

#define ADD_ACTION(container, action, ...) \
  container.push_back(std::unique_ptr<Action>(new action(__VA_ARGS__)))

// create rules for ops that don't change the data
std::unique_ptr<SelectorAndAction> DropQDQNodesRules() {
  // 3 nodes. 0=DQ, 1=target, 2=Q. Merge into target and remove DQ and Q.
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQDropDQDNodesSelector());

  // Move DQ input 0 to target input 0.
  // Move Q output 0 to other output 0.
  std::initializer_list<NodeAndMoveInfo> moves{
      // from input node 0 (DQ) move input value 0 to input 0 of the target node
      MoveToSlot(dq, Direction::kInput, 0, Direction::kInput, 0),
      // from output node 0 (Q) move output value 0 to output 0 of the target node
      MoveToSlot(q, Direction::kOutput, 0, Direction::kOutput, 0)};

  // Delete DQ and Q but not the target node
  NTO::NodeIndexes nodes_to_remove{{0}, false, {0}};

  std::unique_ptr<Action> action(new MergeIntoExisting(moves, nodes_to_remove));

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Gather", {}},
                                                                              {"Reshape", {}},
                                                                              {"Transpose", {}},
                                                                              {"MaxPool", {12}}},
                                             std::move(selector),
                                             std::move(action));
}

std::unique_ptr<SelectorAndAction> BinaryOpQDQRules() {
  // 4 nodes. 0=DQ for inputA, 1=DQ for inputB, 2=target, 3=Q
  // Replace with QLinear version of operator. Delete all original nodes.
  NTO::NodeLocation dq1{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq2{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQBinarySelector());

  std::vector<std::unique_ptr<Action>> actions;

  // set the version point on the two DQ input nodes and the Q output node
  ADD_ACTION(actions, QDQ::SetOptionalZeroPoint, {dq1, dq2, q});
  ADD_ACTION(actions, QDQ::ReplaceWithQLinear,
             kMSDomain,
             {MoveAll(dq1, Direction::kInput),                             // append all inputs from dq1
              MoveAll(dq2, Direction::kInput),                             // append all inputs from dq2
              MoveAndAppend(q, Direction::kInput, 1, Direction::kInput),   // append scale (input 1) from q
              MoveAndAppend(q, Direction::kInput, 2, Direction ::kInput),  // append zp (input 2) from q
              MoveAll(q, Direction::kOutput)});                            // and use the outputs from q

  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Add", {}},
                                                                              {"Mul", {}}},
                                             std::move(selector),
                                             std::move(all_actions));
}

std::unique_ptr<SelectorAndAction> UnaryOpQDQRules() {
  // 3 nodes. 0=DQ, 1=target, 2=Q
  // Replace with QLinear version of operator. Delete all original nodes.
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQUnarySelector());

  std::vector<std::unique_ptr<Action>> actions;
  ADD_ACTION(actions, QDQ::SetOptionalZeroPoint, {dq, q});  // update the DQ and Q nodes
  ADD_ACTION(actions, QDQ::ReplaceWithQLinear,
             kMSDomain,
             {MoveAll(dq, Direction::kInput),                              // append all inputs from dq
              MoveAndAppend(q, Direction::kInput, 1, Direction::kInput),   // append scale (input 1) from q
              MoveAndAppend(q, Direction::kInput, 2, Direction ::kInput),  // append zp (input 2) from q
              MoveAll(q, Direction::kOutput)});                            // and use the outputs from q

  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"AveragePool", {}}},
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
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::unique_ptr<NodeSelector> selector(new QDQConvSelector());

  std::vector<std::unique_ptr<Action>> actions;
  ADD_ACTION(actions, QDQ::ReplaceWithQLinear,
             kOnnxDomain,                                                  // QLinearConv is from ONNX
             {MoveAll(dq_x, Direction::kInput),                            // append all inputs from x
              MoveAll(dq_w, Direction::kInput),                            // append all inputs from w
              MoveAndAppend(q, Direction::kInput, 1, Direction::kInput),   // append scale (input 1) from q
              MoveAndAppend(q, Direction::kInput, 2, Direction ::kInput),  // append zp (input 2) from q
              MoveAll(dq_bias, Direction::kInput),                         // (optional) append input bias
              MoveAll(q, Direction::kOutput)});                            // and use the outputs from q

  // MoveInput(2, 0, -1),    // (optional) append input dq[2][0]
  // MoveOutput(4, -1, -1)}  // and use the outputs from q[0]

  std::unique_ptr<Action> all_actions{new MultiAction{std::move(actions)}};

  return std::make_unique<SelectorAndAction>(SelectorAndAction::OpVersionsMap{{"Conv", {}}},
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
  qdq_selector_action_entries.push_back(std::move(ConvQDQRules()));

  return qdq_selector_action_entries;
}

}  // namespace

QDQSelectorActionTransformer::QDQSelectorActionTransformer()
    : SelectorActionTransformer{
          "QDQSelectorActionTransformer",
          CreateQDQSelectorActionEntries()} {
}

}  // namespace onnxruntime
