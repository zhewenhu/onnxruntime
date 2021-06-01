// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/utils.h"
#include "core/session/inference_session.h"

#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/providers/internal_testing/internal_testing_execution_provider.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {

namespace test {

//// TODO: Refactor cut-and-paste of this so it can be re-used
//static void CreateSession(const SessionOptions& so, std::unique_ptr<InferenceSessionWrapper>& session,
//                          const ORTCHAR_T* model_path = ORT_TSTR("testdata/mnist.onnx"),  // arbitrary test model
//                          bool enable_custom_ep = true,
//                          const std::unordered_set<std::string>* override_supported_ops = nullptr) {
//  session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());
//
//  // set supported ops to ops that are ideally found consecutively in the model.
//  // we can say the EP potentially handles them all, but can also test removing handling of one or more ops
//  // at runtime to simulate a lower spec device where not all ops can be handled. this allows us to test
//  // that we can revert ops back to the CPU implementation successfully
//  const std::unordered_set<std::string> default_supported_ops{"Conv", "Add", "Relu", "MaxPool"};
//  const std::unordered_set<std::string>* supported_ops = override_supported_ops ? override_supported_ops
//                                                                                : &default_supported_ops;
//
//  if (enable_custom_ep) {
//    ASSERT_STATUS_OK(session->RegisterExecutionProvider(
//        std::make_unique<InternalTestingExecutionProvider>(*supported_ops)));
//  }
//
//  ASSERT_STATUS_OK(session->Load(model_path));
//  ASSERT_STATUS_OK(session->Initialize());
//}

TEST(InternalTestingEP, TestMergePartitions) {
  const ORTCHAR_T* model_path = ORT_TSTR("testdata/merge_partitions.onnx");

  // make sure we can't save a model with compiled ops. input/output model format doesn't matter
  SessionOptions so;
  auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

  const std::unordered_set<std::string> supported_ops{"Add"};
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(
      std::make_unique<InternalTestingExecutionProvider>(supported_ops)));

  ASSERT_STATUS_OK(session->Load(model_path));
  const auto& graph = session->GetGraph();
  GraphViewer viewer{graph};

  InternalTestingExecutionProvider ep{supported_ops};
  auto result = ep.GetCapability(viewer, {});
  auto result2 = ep.GetCapability2(viewer, {});

  ASSERT_STATUS_OK(session->Initialize());

  const auto& func_mgr = session->GetSessionState().GetFuncMgr();
  NodeComputeInfo* compute_func = nullptr;

  for (const auto& node : graph.Nodes()) {
    EXPECT_EQ(supported_ops.count(node.OpType()), size_t(0))
        << "Nodes with supported op types should have been replaced. Node with type " << node.OpType() << " was not.";
    if (node.GetExecutionProviderType() == utils::kInternalTestingExecutionProvider) {
      EXPECT_STATUS_OK(func_mgr.GetFuncs(node.Name(), compute_func));
      EXPECT_NE(compute_func, nullptr);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
