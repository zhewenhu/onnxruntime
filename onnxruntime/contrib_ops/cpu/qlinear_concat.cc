// Copyright (c Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_concat.h"
#include "contrib_ops/cpu/qlinear_lookup_table.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

QLinearConcat::QLinearConcat(const OpKernelInfo& info) : OpKernel(info), ConcatBase(info) {
  size_t input_def_count = info.node().InputDefs().size();
  ORT_ENFORCE(input_def_count >= 8 && (input_def_count - 2) % 3 == 0,
              "At lease two inputs are needed, and each input must be (tensor, scale, zero_point) tuple!");

#if defined(QLCONCAT_TABLE)

  size_t input_count = (input_def_count - 2) / 3;
  fixed_lookup_tables_.resize(input_count);

  const Tensor* tensor_y_scale = nullptr;
  const Tensor* tensor_y_zero_point = nullptr;
  bool get_y_scale = info.TryGetConstantInput(0, &tensor_y_scale);
  bool get_y_zero_point = info.TryGetConstantInput(1, &tensor_y_zero_point);
  if (!get_y_scale || !get_y_zero_point) {
    // Can not build const lookup table
    return;
  }

  // Initialize lookup table given constant input/output scales and zero points
  bool is_signed_int8 = tensor_y_zero_point->IsDataType<int8_t>();
  const auto identity_float = [](float v) -> float { return v; };
  for (size_t def_index = 2; def_index < input_def_count; def_index += 3) {
    const Tensor* tensor_x_scale = nullptr;
    const Tensor* tensor_x_zero_point = nullptr;
    bool get_x_scale = info.TryGetConstantInput(static_cast<int>(def_index) + 1, &tensor_x_scale);
    bool get_x_zero_point = !info.TryGetConstantInput(static_cast<int>(def_index) + 2, &tensor_x_zero_point);
    if (!get_x_scale || !get_x_zero_point) {
      continue;  // try to optimize next one
    }
    ORT_ENFORCE(tensor_x_scale->IsDataType<float>(), "Input scale is not float for input def @", def_index + 1);
    ORT_ENFORCE(tensor_x_zero_point->GetElementType() == tensor_y_zero_point->GetElementType(),
                "Wrong input type encountered for zero point input def @", def_index + 2);

    size_t input_idx = (def_index - 2) / 3;
    fixed_lookup_tables_[input_idx].resize(256);
    if (is_signed_int8) {
      QlinearBuildLookupTable<int8_t>(
          fixed_lookup_tables_[input_idx].data(), tensor_x_scale, tensor_x_zero_point,
          tensor_y_scale, tensor_y_zero_point, identity_float);
    } else {
      QlinearBuildLookupTable<uint8_t>(
          fixed_lookup_tables_[input_idx].data(), tensor_x_scale, tensor_x_zero_point,
          tensor_y_scale, tensor_y_zero_point, identity_float);
    }
  }

#endif

}

#if defined(QLCONCAT_TABLE)

Status QLinearConcat::Compute(OpKernelContext* ctx) const {
  const Tensor* tensor_y_scale = ctx->Input<Tensor>(0);
  const Tensor* tensor_y_zero_point = ctx->Input<Tensor>(1);
  bool is_signed_int8 = tensor_y_zero_point->IsDataType<int8_t>();
  const auto identity_float = [](float v) -> float { return v; };

  // Number of input tensors to concatenate (tupled)
  auto input_count_x3 = Node().InputArgCount()[2];
  ORT_ENFORCE(input_count_x3 >= 6 && input_count_x3 % 3 == 0,
              "At lease two inputs are needed, and each input must be (tensor, scale, zero_point) tuple!");

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  auto input_count = input_count_x3 / 3;
  std::vector<std::vector<uint8_t>> dynamic_lookup_tables(input_count);

  std::vector<const Tensor*> input_tensors(input_count);
  for (auto input_index = 0; input_index < input_count; ++input_index) {
    auto tuple_start = 2 + input_index * 3;
    input_tensors[input_index] = ctx->Input<Tensor>(static_cast<int>(tuple_start));

    // Build lookup table for non-const parameters (scale + zero point)
    if (fixed_lookup_tables_[input_index].size() == 0) {
      // Check tensor type first
      const Tensor* tensor_x_scale = ctx->Input<Tensor>(tuple_start + 1);
      const Tensor* tensor_x_zero_point = ctx->Input<Tensor>(tuple_start + 2);

      ORT_ENFORCE(tensor_x_scale->IsDataType<float>(), "Input scale is not float for quantized input @", tuple_start + 1);
      ORT_ENFORCE(tensor_x_zero_point->GetElementType() == tensor_y_zero_point->GetElementType(),
                  "Wrong input type encountered for zero point of quantized input @", tuple_start + 2);

      dynamic_lookup_tables[input_index].resize(256);
      if (is_signed_int8) {
        QlinearBuildLookupTable<int8_t>(
            dynamic_lookup_tables[input_index].data(), tensor_x_scale, tensor_x_zero_point,
            tensor_y_scale, tensor_y_zero_point, identity_float);
      } else {
        QlinearBuildLookupTable<uint8_t>(
            dynamic_lookup_tables[input_index].data(), tensor_x_scale, tensor_x_zero_point,
            tensor_y_scale, tensor_y_zero_point, identity_float);
      }
    }
  }

  // Validate inputs and prepare some metadata used during actual compute
  Prepare p;
  auto status = PrepareForCompute(ctx, input_tensors, p);
  if (!status.IsOK())
    return status;

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  int64_t initial_output_offset = 0;  // initial offset for each input
  for (int input_index = 0; input_index < input_count; input_index++) {
    const auto& prep = p.inputs[input_index];
    if (prep.num_elements == 0)
      continue;

    auto input_axis_pitch = prep.axis_pitch;
    const uint8_t* input = static_cast<const uint8_t*>(prep.tensor->DataRaw());

    auto input_size = prep.num_elements;

    uint8_t* output = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw());
    int64_t cur_in_offset = 0;
    int64_t cur_out_offset = initial_output_offset;
    for (size_t idx_copy = 0, end = input_size / input_axis_pitch; idx_copy < end; ++idx_copy) {
      const uint8_t* table = (fixed_lookup_tables_[input_index].size() > 0)
                                 ? fixed_lookup_tables_[input_index].data()
                                 : dynamic_lookup_tables[input_index].data();
      QLinearLookupTableTransform(input + cur_in_offset, table, output + cur_out_offset, input_axis_pitch);

      cur_out_offset += p.output_axis_pitch;
      cur_in_offset += input_axis_pitch;
    }

    initial_output_offset += input_axis_pitch;
  }

  return Status::OK();
}

#else

Status QLinearConcat::Compute(OpKernelContext* ctx) const {
  const Tensor* tensor_y_scale = ctx->Input<Tensor>(0);
  const Tensor* tensor_y_zero_point = ctx->Input<Tensor>(1);
  bool is_signed_int8 = tensor_y_zero_point->IsDataType<int8_t>();

  // Number of input tensors to concatenate (tupled)
  auto input_count_x3 = Node().InputArgCount()[2];
  ORT_ENFORCE(input_count_x3 >= 6 && input_count_x3 % 3 == 0,
              "At lease two inputs are needed, and each input must be (tensor, scale, zero_point) tuple!");

  auto input_count = input_count_x3 / 3;
  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  std::vector<const Tensor*> input_tensors(input_count);
  std::vector<float> input_scales(input_count);
  std::vector<int> input_zero_points(input_count);
  float scale_y = *tensor_y_scale->Data<float>();

  for (auto input_index = 0; input_index < input_count; ++input_index) {
    auto tuple_start = 2 + input_index * 3;
    input_tensors[input_index] = ctx->Input<Tensor>(static_cast<int>(tuple_start));

    const Tensor* tensor_x_scale = ctx->Input<Tensor>(tuple_start + 1);
    const Tensor* tensor_x_zero_point = ctx->Input<Tensor>(tuple_start + 2);

    ORT_ENFORCE(tensor_x_scale->IsDataType<float>(), "Input scale is not float for quantized input @", tuple_start + 1);
    input_scales[input_index] = *tensor_x_scale->Data<float>();

    ORT_ENFORCE(tensor_x_zero_point->GetElementType() == tensor_y_zero_point->GetElementType(),
                "Wrong input type encountered for zero point of quantized input @", tuple_start + 2);
    if (is_signed_int8) {
      input_zero_points[input_index] = *tensor_x_zero_point->Data<int8_t>();
    } else {
      input_zero_points[input_index] = *tensor_x_zero_point->Data<uint8_t>();
    }
  }

  // Validate inputs and prepare some metadata used during actual compute
  Prepare p;
  auto status = PrepareForCompute(ctx, input_tensors, p);
  if (!status.IsOK())
    return status;

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  int64_t initial_output_offset = 0;  // initial offset for each input
  for (int input_index = 0; input_index < input_count; input_index++) {
    const auto& prep = p.inputs[input_index];
    if (prep.num_elements == 0)
      continue;

    auto input_axis_pitch = prep.axis_pitch;
    auto input_size = prep.num_elements;
    int64_t cur_in_offset = 0;
    int64_t cur_out_offset = initial_output_offset;
    if (is_signed_int8) {
      const int8_t* input = prep.tensor->Data<int8_t>();
      int8_t* output = p.output_tensor->MutableData<int8_t>();
      for (size_t idx_copy = 0, end = input_size / input_axis_pitch; idx_copy < end; ++idx_copy) {
        MlasRequantizeLinear<int8_t>(input + cur_in_offset, input_scales[input_index], input_zero_points[input_index],
                             output + cur_out_offset, scale_y, (int32_t)*tensor_y_zero_point->Data<int8_t>(),
                             input_axis_pitch);
        cur_out_offset += p.output_axis_pitch;
        cur_in_offset += input_axis_pitch;
      }
    } else {
      const uint8_t* input = prep.tensor->Data<uint8_t>();
      uint8_t* output = p.output_tensor->MutableData<uint8_t>();
      for (size_t idx_copy = 0, end = input_size / input_axis_pitch; idx_copy < end; ++idx_copy) {
        MlasRequantizeLinear<uint8_t>(input + cur_in_offset, input_scales[input_index], input_zero_points[input_index],
                             output + cur_out_offset, scale_y, (int32_t)*tensor_y_zero_point->Data<uint8_t>(),
                             input_axis_pitch);
        cur_out_offset += p.output_axis_pitch;
        cur_in_offset += input_axis_pitch;
      }
    }

    initial_output_offset += input_axis_pitch;
  }

  return Status::OK();
}

#endif

ONNX_OPERATOR_KERNEL_EX(QLinearConcat, kMSDomain, 1, kCpuExecutionProvider, KernelDefBuilder(), QLinearConcat);

}  // namespace contrib
}  // namespace onnxruntime
