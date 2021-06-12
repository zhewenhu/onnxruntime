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

Transformer that fuses QDQ and fp32 ops into quantized ops 
Using rules to define the nodes to match and actions to take.
*/
class RulesTransformer : public GraphTransformer {
 public:
  RulesTransformer() noexcept;

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  // on set of rules, keyed on OpType. Domain is checked within Rules given that should rarely clash.
  const std::unordered_map<std::string, Rules> rules_;
};

}  // namespace QDQ
}  // namespace onnxruntime
