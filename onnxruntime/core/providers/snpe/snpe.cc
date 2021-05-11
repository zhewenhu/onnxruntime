// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "snpe.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    Snpe,
    kMSDomain,
    1,
    kSnpeExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Snpe);

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
