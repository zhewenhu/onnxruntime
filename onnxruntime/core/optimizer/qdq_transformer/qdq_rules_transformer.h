// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/qdq_transformer/transformer_rules.h"

namespace onnxruntime {
namespace QDQ {
/**
@Class QDQ::RulesTransformer

Transformer that fuses QDQ and fp32 ops into quantized ops. 
Uses rules to define the nodes to match and actions to take.
*/
class RulesTransformer : public GraphTransformer {
 public:
  RulesTransformer() noexcept;

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  std::vector<std::unique_ptr<SelectorAndActions>> rules_and_actions_;
  std::unordered_map<std::string, const SelectorAndActions*> op_to_rules_and_actions_;
};

}  // namespace QDQ
}  // namespace onnxruntime
