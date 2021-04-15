/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize.cpp

Abstract:

    This module implements routines to quantize buffers.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "mlasi.h"

#if defined(MLAS_NEON64_INTRINSICS) || defined(MLAS_SSE2_INTRINSICS)

//
// QuantizeLinear implementation using NEON or SSE2 intrinsics.
//

MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearVector(
    MLAS_FLOAT32X4 FloatVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_FLOAT32X4 MinimumValueVector,
    MLAS_FLOAT32X4 MaximumValueVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
    //
    // Scale the input vector and clamp the values to the minimum and maximum
    // range (adjusted by the zero point value).
    //

    FloatVector = MlasDivideFloat32x4(FloatVector, ScaleVector);

#if defined(MLAS_NEON64_INTRINSICS)
    // N.B. FMINNM and FMAXNM returns the numeric value if either of the values
    // is a NaN.
    FloatVector = vmaxnmq_f32(FloatVector, MinimumValueVector);
    FloatVector = vminnmq_f32(FloatVector, MaximumValueVector);
#else
    // N.B. MINPS and MAXPS returns the value from the second vector if the
    // value from the first vector is a NaN.
    FloatVector = _mm_max_ps(FloatVector, MinimumValueVector);
    FloatVector = _mm_min_ps(FloatVector, MaximumValueVector);
#endif

    //
    // Convert the float values to integer using "round to nearest even" and
    // then shift the output range using the zero point value.
    //

#if defined(MLAS_NEON64_INTRINSICS)
    auto IntegerVector = vcvtnq_s32_f32(FloatVector);
    IntegerVector = vaddq_s32(IntegerVector, ZeroPointVector);
#else
    // N.B. Assumes MXCSR has been configured with the default rounding mode of
    // "round to nearest even".
    auto IntegerVector = _mm_cvtps_epi32(FloatVector);
    IntegerVector = _mm_add_epi32(IntegerVector, ZeroPointVector);
#endif

    return IntegerVector;
}

template<typename OutputType>
MLAS_INT32X4
MlasQuantizeLinearPackBytes(
    MLAS_INT32X4 IntegerVector
    );

#if defined(MLAS_NEON64_INTRINSICS)

template<typename OutputType>
MLAS_INT32X4
MlasQuantizeLinearPackBytes(
    MLAS_INT32X4 IntegerVector
    )
{
    //
    // Swizzle the least significant byte from each int32_t element to the
    // bottom four bytes of the vector register.
    //

    uint16x8_t WordVector = vreinterpretq_u16_s32(IntegerVector);
    WordVector = vuzp1q_u16(WordVector, WordVector);
    uint8x16_t ByteVector = vreinterpretq_u8_u16(WordVector);
    ByteVector = vuzp1q_u8(ByteVector, ByteVector);

    return vreinterpretq_s32_u8(ByteVector);
}

#elif defined(MLAS_NEON_INTRINSICS)

template <>
MLAS_INT32X4
MlasQuantizeLinearPackBytes<int8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    const auto s16x4 = vmovn_s32(IntegerVector);
    const auto s16x8 = vcombine(s16x4, s16x4);
    const auto s8x8 = vmovn_s16(s16x8);
    const auto s8x16 = vcombine_s8(s8x8, s8x8);
    return vreinterpretq_s32_s8(ByteVector);
}

template <>
MLAS_INT32X4
MlasQuantizeLinearPackBytes<uint8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    const auto s16x4 = vmovn_s32(IntegerVector);
    const auto s16x8 = vcombine(s16x4, s16x4);
    const auto u8x8 = vmovun_s16(s16x8);
    const auto u8x16 = vcombine_u8(u8x8, u8x8);
    return vreinterpretq_s32_u8(ByteVector);
}

#else

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearPackBytes<uint8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

    return IntegerVector;
}

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearPackBytes<int8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    IntegerVector = _mm_packs_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_packs_epi16(IntegerVector, IntegerVector);

    return IntegerVector;
}

#endif

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinearKernel(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    auto ScaleVector = MlasBroadcastFloat32x4(Scale);
    auto MinimumValueVector = MlasBroadcastFloat32x4(float(MinimumValue - ZeroPoint));
    auto MaximumValueVector = MlasBroadcastFloat32x4(float(MaximumValue - ZeroPoint));
    auto ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);

    while (N >= 4) {

        auto FloatVector = MlasLoadFloat32x4(Input);
        auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
            MinimumValueVector, MaximumValueVector, ZeroPointVector);

        IntegerVector = MlasQuantizeLinearPackBytes<OutputType>(IntegerVector);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_s32((int32_t*)Output, IntegerVector, 0);
#else
        *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);
#endif

        Input += 4;
        Output += 4;
        N -= 4;
    }

    for (size_t n = 0; n < N; n++) {

#if defined(MLAS_NEON64_INTRINSICS)
        auto FloatVector = vld1q_dup_f32(Input + n);
#else
        auto FloatVector = _mm_load_ss(Input + n);
#endif
        auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
            MinimumValueVector, MaximumValueVector, ZeroPointVector);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_u8((uint8_t*)Output + n, vreinterpretq_u8_s32(IntegerVector), 0);
#else
        *((uint8_t*)Output + n) = (uint8_t)_mm_cvtsi128_si32(IntegerVector);
#endif
    }
}

void
MLASCALL
MlasQuantizeLinearS8Kernel(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    )
{
    MlasQuantizeLinearKernel<int8_t>(Input, Output, N, Scale, ZeroPoint);
}

void
MLASCALL
MlasQuantizeLinearU8Kernel(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
)
{
    MlasQuantizeLinearKernel<uint8_t>(Input, Output, N, Scale, ZeroPoint);
}

template<>
void
MLASCALL
MlasQuantizeLinear<int8_t>(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    )
{
#if defined(MLAS_TARGET_AMD64)
    MlasPlatform.QuantizeLinearS8Kernel(
#else
    MlasQuantizeLinearS8Kernel(
#endif
        Input, Output, N, Scale, ZeroPoint);
}

template<>
void
MLASCALL
MlasQuantizeLinear<uint8_t>(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
{
#if defined(MLAS_TARGET_AMD64)
    MlasPlatform.QuantizeLinearU8Kernel(
#else
    MlasQuantizeLinearU8Kernel(
#endif
        Input, Output, N, Scale, ZeroPoint);
}

#else

//
// QuantizeLinear implementation using the C++ runtime.
//

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    for (size_t n = 0; n < N; n++) {

        float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
        FloatValue = std::max(FloatValue, float(MinimumValue));
        FloatValue = std::min(FloatValue, float(MaximumValue));
        Output[n] = (OutputType)(int32_t)FloatValue;
    }
}

template
void
MLASCALL
MlasQuantizeLinear<int8_t>(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    );

template
void
MLASCALL
MlasQuantizeLinear<uint8_t>(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    );
#endif

#if defined(MLAS_SSE2_INTRINSICS)

void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    const float* Scale,
    bool PerColumnScale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale.

    PerColumnScale - Supplies true if the quantization scale has per-column
        values, else false if a single quantization scale applies to the
        entire matrix.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    const __m128 PerMatrixScaleVector = PerColumnScale ? _mm_setzero_ps() : _mm_load1_ps(Scale);
    const __m128 MinimumValueVector = _mm_set1_ps(float(0 - ZeroPoint));
    const __m128 MaximumValueVector = _mm_set1_ps(float(255 - ZeroPoint));
    const __m128i ZeroPointVector = _mm_set1_epi32(ZeroPoint);

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        const int32_t* bias = Bias;
        const float* scale = PerColumnScale ? Scale : nullptr;
        size_t n = N;

        //
        // Process 16 columns of the matrices at a time.
        //

        while (n >= 16) {

            //
            // Load the input data and optionally add the per-column bias.
            //

            __m128i IntegerVector0 = _mm_loadu_si128((const __m128i *)&Input[0]);
            __m128i IntegerVector1 = _mm_loadu_si128((const __m128i *)&Input[4]);
            __m128i IntegerVector2 = _mm_loadu_si128((const __m128i *)&Input[8]);
            __m128i IntegerVector3 = _mm_loadu_si128((const __m128i *)&Input[12]);
            Input += 16;

            if (bias != nullptr) {
                IntegerVector0 = _mm_add_epi32(IntegerVector0, _mm_loadu_si128((const __m128i *)&bias[0]));
                IntegerVector1 = _mm_add_epi32(IntegerVector1, _mm_loadu_si128((const __m128i *)&bias[4]));
                IntegerVector2 = _mm_add_epi32(IntegerVector2, _mm_loadu_si128((const __m128i *)&bias[8]));
                IntegerVector3 = _mm_add_epi32(IntegerVector3, _mm_loadu_si128((const __m128i *)&bias[12]));
                bias += 16;
            }

            //
            // Convert to integer values to float and apply the per-tensor or
            // per-column scaling.
            //

            __m128 FloatVector0 = _mm_cvtepi32_ps(IntegerVector0);
            __m128 FloatVector1 = _mm_cvtepi32_ps(IntegerVector1);
            __m128 FloatVector2 = _mm_cvtepi32_ps(IntegerVector2);
            __m128 FloatVector3 = _mm_cvtepi32_ps(IntegerVector3);

            if (scale != nullptr) {

                FloatVector0 = _mm_mul_ps(FloatVector0, _mm_loadu_ps(&scale[0]));
                FloatVector1 = _mm_mul_ps(FloatVector1, _mm_loadu_ps(&scale[4]));
                FloatVector2 = _mm_mul_ps(FloatVector2, _mm_loadu_ps(&scale[8]));
                FloatVector3 = _mm_mul_ps(FloatVector3, _mm_loadu_ps(&scale[12]));
                scale += 16;

            } else {

                FloatVector0 = _mm_mul_ps(FloatVector0, PerMatrixScaleVector);
                FloatVector1 = _mm_mul_ps(FloatVector1, PerMatrixScaleVector);
                FloatVector2 = _mm_mul_ps(FloatVector2, PerMatrixScaleVector);
                FloatVector3 = _mm_mul_ps(FloatVector3, PerMatrixScaleVector);
            }

            FloatVector0 = _mm_max_ps(FloatVector0, MinimumValueVector);
            FloatVector1 = _mm_max_ps(FloatVector1, MinimumValueVector);
            FloatVector2 = _mm_max_ps(FloatVector2, MinimumValueVector);
            FloatVector3 = _mm_max_ps(FloatVector3, MinimumValueVector);

            FloatVector0 = _mm_min_ps(FloatVector0, MaximumValueVector);
            FloatVector1 = _mm_min_ps(FloatVector1, MaximumValueVector);
            FloatVector2 = _mm_min_ps(FloatVector2, MaximumValueVector);
            FloatVector3 = _mm_min_ps(FloatVector3, MaximumValueVector);

            IntegerVector0 = _mm_cvtps_epi32(FloatVector0);
            IntegerVector1 = _mm_cvtps_epi32(FloatVector1);
            IntegerVector2 = _mm_cvtps_epi32(FloatVector2);
            IntegerVector3 = _mm_cvtps_epi32(FloatVector3);

            IntegerVector0 = _mm_add_epi32(IntegerVector0, ZeroPointVector);
            IntegerVector1 = _mm_add_epi32(IntegerVector1, ZeroPointVector);
            IntegerVector2 = _mm_add_epi32(IntegerVector2, ZeroPointVector);
            IntegerVector3 = _mm_add_epi32(IntegerVector3, ZeroPointVector);

            __m128i WordVector0 = _mm_packus_epi16(IntegerVector0, IntegerVector1);
            __m128i WordVector1 = _mm_packus_epi16(IntegerVector2, IntegerVector3);

            __m128i ByteVector = _mm_packus_epi16(WordVector0, WordVector1);

            _mm_storeu_si128((__m128i*)Output, ByteVector);
            Output += 16;

            n -= 16;
        }

        //
        // Process the remaining columns of the matrices.
        //

        while (n > 0) {

            //
            // Load the input data and optionally add the per-column bias.
            //

            __m128i IntegerVector;

            if (n >= 4) {

                IntegerVector = _mm_loadu_si128((const __m128i*)&Input[0]);
                Input += 4;

                if (bias != nullptr) {
                    IntegerVector = _mm_add_epi32(IntegerVector, _mm_loadu_si128((const __m128i*)&bias[0]));
                    bias += 4;
                }

            } else {

                int32_t IntegerValue = *Input++;

                if (bias != nullptr) {
                    IntegerValue += *bias++;
                }

                IntegerVector = _mm_cvtsi32_si128(IntegerValue);
            }

            //
            // Convert to integer values to float and apply the per-tensor or
            // per-column scaling.
            //

            __m128 FloatVector = _mm_cvtepi32_ps(IntegerVector);
            __m128 ScaleVector;

            if (scale != nullptr) {

                if (n >= 4) {
                    ScaleVector = _mm_loadu_ps(scale);
                    scale += 4;
                } else {
                    ScaleVector = _mm_load_ss(scale);
                    scale += 1;
                }

            } else {
                ScaleVector = PerMatrixScaleVector;
            }

            FloatVector = _mm_mul_ps(FloatVector, ScaleVector);

            FloatVector = _mm_max_ps(FloatVector, MinimumValueVector);
            FloatVector = _mm_min_ps(FloatVector, MaximumValueVector);

            IntegerVector = _mm_cvtps_epi32(FloatVector);
            IntegerVector = _mm_add_epi32(IntegerVector, ZeroPointVector);

            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

            uint32_t OutputValue = uint32_t(_mm_cvtsi128_si32(IntegerVector));

            if (n >= 4) {

                *reinterpret_cast<uint32_t*>(Output) = OutputValue;
                Output += 4;

                n -= 4;

            } else {

                *Output = uint8_t(OutputValue);
                Output += 1;

                n -= 1;
            }
        }
    }
}

#elif defined(MLAS_NEON64_INTRINSICS)

void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    const float* Scale,
    bool PerColumnScale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale.

    PerColumnScale - Supplies true if the quantization scale has per-column
        values, else false if a single quantization scale applies to the
        entire matrix.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    const float32x4_t PerMatrixScaleVector = PerColumnScale ? vdupq_n_f32(0) : vld1q_dup_f32(Scale);
    const int16x8_t ZeroPointVector = vdupq_n_s16(ZeroPoint);

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        const int32_t* bias = Bias;
        const float* scale = PerColumnScale ? Scale : nullptr;
        size_t n = N;

        //
        // Process 16 columns of the matrices at a time.
        //

        while (n >= 16) {

            //
            // Load the input data and optionally add the per-column bias.
            //

            int32x4x4_t IntegerVector;

            IntegerVector.val[0] = vld1q_s32(&Input[0]);
            IntegerVector.val[1] = vld1q_s32(&Input[4]);
            IntegerVector.val[2] = vld1q_s32(&Input[8]);
            IntegerVector.val[3] = vld1q_s32(&Input[12]);
            Input += 16;

            if (bias != nullptr) {
                IntegerVector.val[0] = vaddq_s32(IntegerVector.val[0], vld1q_s32(&bias[0]));
                IntegerVector.val[1] = vaddq_s32(IntegerVector.val[1], vld1q_s32(&bias[4]));
                IntegerVector.val[2] = vaddq_s32(IntegerVector.val[2], vld1q_s32(&bias[8]));
                IntegerVector.val[3] = vaddq_s32(IntegerVector.val[3], vld1q_s32(&bias[12]));
                bias += 16;
            }

            //
            // Convert to integer values to float and apply the per-tensor or
            // per-column scaling.
            //

            float32x4x4_t FloatVector;

            FloatVector.val[0] = vcvtq_f32_s32(IntegerVector.val[0]);
            FloatVector.val[1] = vcvtq_f32_s32(IntegerVector.val[1]);
            FloatVector.val[2] = vcvtq_f32_s32(IntegerVector.val[2]);
            FloatVector.val[3] = vcvtq_f32_s32(IntegerVector.val[3]);

            if (scale != nullptr) {

                float32x4x4_t PerColumnScaleVector;

                PerColumnScaleVector.val[0] = vld1q_f32(&scale[0]);
                PerColumnScaleVector.val[1] = vld1q_f32(&scale[4]);
                PerColumnScaleVector.val[2] = vld1q_f32(&scale[8]);
                PerColumnScaleVector.val[3] = vld1q_f32(&scale[12]);
                scale += 16;

                FloatVector.val[0] = vmulq_f32(FloatVector.val[0], PerColumnScaleVector.val[0]);
                FloatVector.val[1] = vmulq_f32(FloatVector.val[1], PerColumnScaleVector.val[1]);
                FloatVector.val[2] = vmulq_f32(FloatVector.val[2], PerColumnScaleVector.val[2]);
                FloatVector.val[3] = vmulq_f32(FloatVector.val[3], PerColumnScaleVector.val[3]);

            } else {

                FloatVector.val[0] = vmulq_f32(FloatVector.val[0], PerMatrixScaleVector);
                FloatVector.val[1] = vmulq_f32(FloatVector.val[1], PerMatrixScaleVector);
                FloatVector.val[2] = vmulq_f32(FloatVector.val[2], PerMatrixScaleVector);
                FloatVector.val[3] = vmulq_f32(FloatVector.val[3], PerMatrixScaleVector);
            }

            //
            // Convert the float values to integer using "round to nearest even".
            // Results are saturated to the range of int32_t.
            //

            IntegerVector.val[0] = vcvtnq_s32_f32(FloatVector.val[0]);
            IntegerVector.val[1] = vcvtnq_s32_f32(FloatVector.val[1]);
            IntegerVector.val[2] = vcvtnq_s32_f32(FloatVector.val[2]);
            IntegerVector.val[3] = vcvtnq_s32_f32(FloatVector.val[3]);

            //
            // Pack the integers with saturation to 16-bit values and shift by
            // the zero point, then pack the integers again to unsigned bytes.
            //

            int16x8x2_t WordVector;

            WordVector.val[0] = vqmovn_high_s32(vqmovn_s32(IntegerVector.val[0]), IntegerVector.val[1]);
            WordVector.val[1] = vqmovn_high_s32(vqmovn_s32(IntegerVector.val[2]), IntegerVector.val[3]);

            WordVector.val[0] = vqaddq_s16(WordVector.val[0], ZeroPointVector);
            WordVector.val[1] = vqaddq_s16(WordVector.val[1], ZeroPointVector);

            vst1q_u8(Output, vqmovun_high_s16(vqmovun_s16(WordVector.val[0]), WordVector.val[1]));
            Output += 16;

            n -= 16;
        }

        //
        // Process the remaining columns of the matrices.
        //

        while (n > 0) {

            //
            // Load the input data and optionally add the per-column bias.
            //

            int32x4_t IntegerVector;

            if (n >= 4) {

                IntegerVector = vld1q_s32(&Input[0]);
                Input += 4;

                if (bias != nullptr) {
                    IntegerVector = vaddq_s32(IntegerVector, vld1q_s32(&bias[0]));
                    bias += 4;
                }

            } else {

                IntegerVector = vld1q_dup_s32(Input);
                Input += 1;

                if (bias != nullptr) {
                    IntegerVector = vaddq_s32(IntegerVector, vld1q_dup_s32(bias));
                    bias += 1;
                }
            }

            //
            // Convert to integer values to float and apply the per-tensor or
            // per-column scaling.
            //

            float32x4_t FloatVector = vcvtq_f32_s32(IntegerVector);
            float32x4_t ScaleVector;

            if (scale != nullptr) {

                if (n >= 4) {
                    ScaleVector = vld1q_f32(scale);
                    scale += 4;
                } else {
                    ScaleVector = vld1q_dup_f32(scale);
                    scale += 1;
                }

            } else {
                ScaleVector = PerMatrixScaleVector;
            }

            FloatVector = vmulq_f32(FloatVector, ScaleVector);

            //
            // Convert the float values to integer using "round to nearest even".
            // Results are saturated to the range of int32_t.
            //

            IntegerVector = vcvtnq_s32_f32(FloatVector);

            //
            // Pack the integers with saturation to 16-bit values and shift by
            // the zero point, then pack the integers again to unsigned bytes.
            //

            int16x8_t WordVector = vcombine_s16(vqmovn_s32(IntegerVector), vdup_n_s16(0));
            WordVector = vqaddq_s16(WordVector, ZeroPointVector);

            uint8x16_t ByteVector = vcombine_u8(vqmovun_s16(WordVector), vdup_n_u8(0));

            if (n >= 4) {

                vst1q_lane_u32(reinterpret_cast<uint32_t*>(Output), vreinterpretq_u32_u8(ByteVector), 0);
                Output += 4;

                n -= 4;

            } else {

                vst1q_lane_u8(Output, ByteVector, 0);
                Output += 1;

                n -= 1;
            }
        }
    }
}

#else

void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    const float* Scale,
    bool PerColumnScale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale.

    PerColumnScale - Supplies true if the quantization scale has per-column
        values, else false if a single quantization scale applies to the
        entire matrix.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    const float PerMatrixScaleValue = PerColumnScale ? 0.0f : *Scale;
    const float MinimumValue = float(0 - ZeroPoint);
    const float MaximumValue = float(255 - ZeroPoint);

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        const int32_t* bias = Bias;
        const float* scale = Scale;
        size_t n = N;

        while (n > 0) {

            int32_t IntegerValue = *Input++;

            if (bias != nullptr) {
                IntegerValue += *bias++;
            }

            float FloatValue = float(IntegerValue);
            float ScaleValue = PerColumnScale ? *scale++ : PerMatrixScaleValue;

            FloatValue *= ScaleValue;
            FloatValue = std::max(FloatValue, MinimumValue);
            FloatValue = std::min(FloatValue, MaximumValue);

            //
            // Use the fast rounding trick adapted from XNNPACK: bias the floating
            // point value by the first floating point value that has no
            // fractional bits. The add operation performs the "round to nearest
            // even". Extract the mantissa bits from this floating point value to
            // obtain the rounded integer value.
            //

            IntegerValue = int32_t(MlasBitsOfFp32(FloatValue + MLAS_ROUNDING_BIAS_MAGIC)) -
                MLAS_ROUNDING_BIAS_MAGIC_BITS;

            *Output++ = uint8_t(IntegerValue + ZeroPoint);

            n -= 1;
        }
    }
}

#endif

void
MLASCALL
MlasFindMinMaxElement(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
    )
/*++

Routine Description:

    This routine finds the minimum and maximum values of the supplied buffer.

Arguments:

    Input - Supplies the input buffer.

    Min - Returns the minimum value of the supplied buffer.

    Max - Returns the maximum value of the supplied buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    MlasPlatform.ReduceMinimumMaximumF32Kernel(Input, Min, Max, N);
#else
    MlasReduceMinimumMaximumF32Kernel(Input, Min, Max, N);
#endif
}


#include "qladd.h"

#if defined(MLAS_SSE2_INTRINSICS)

template <typename XInt8>
void
MLASCALL
MlasRequantizeLinear(
    const XInt8* Input,
    float ScaleIn,
    int32_t ZeroPointIn,
    XInt8* Output,
    float ScaleOut,
    int32_t ZeroPointOut,
    size_t N
    )
{
    auto ScaleVector = MlasBroadcastFloat32x4(ScaleOut / ScaleIn);
    auto MinimumValueVector = MlasBroadcastFloat32x4(float((int)std::numeric_limits<XInt8>::min() - ZeroPointOut));
    auto MaximumValueVector = MlasBroadcastFloat32x4(float((int)std::numeric_limits<XInt8>::max() - ZeroPointOut));
    auto ZeroPointVectorOut = MlasBroadcastInt32x4(ZeroPointOut);
    auto ZeroPointVectorIn = MlasBroadcastInt32x4(ZeroPointIn);

    for(;N >= 8; N -= 8) {
        const auto va_i8x8 = _mm_loadl_epi64((const MLAS_INT32X4*)Input);
        const auto va_i16x8 = _mm_unpacklo_epi8(va_i8x8, va_i8x8);
        Input += 8;

        auto va_i32_lo = _mm_sub_epi32(MlasShiftRightInt32<XInt8>(_mm_unpacklo_epi16(va_i16x8, va_i16x8), 24), ZeroPointVectorIn);
        auto va_i32_hi = _mm_sub_epi32(MlasShiftRightInt32<XInt8>(_mm_unpackhi_epi16(va_i16x8, va_i16x8), 24), ZeroPointVectorIn);
        va_i32_lo = MlasQuantizeLinearVector(_mm_cvtepi32_ps(va_i32_lo), ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVectorOut);
        va_i32_hi = MlasQuantizeLinearVector(_mm_cvtepi32_ps(va_i32_hi), ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVectorOut);
        va_i32_lo = MlasQuantizeLinearPackBytes<XInt8>(va_i32_lo);
        va_i32_hi = MlasQuantizeLinearPackBytes<XInt8>(va_i32_hi);

        *((int32_t*)Output) = _mm_cvtsi128_si32(va_i32_lo);
        *((int32_t*)(Output + 4)) = _mm_cvtsi128_si32(va_i32_hi);

        Output += 8;
    }

    if (N > 0) {
        XInt8 tail[8];
        memcpy(tail, Input, N);

        const auto va_i8x8 = _mm_loadl_epi64((const MLAS_INT32X4*)tail);
        const auto va_i16x8 = _mm_unpacklo_epi8(va_i8x8, va_i8x8);

        auto va_i32_lo = _mm_sub_epi32(MlasShiftRightInt32<XInt8>(_mm_unpacklo_epi16(va_i16x8, va_i16x8), 24), ZeroPointVectorIn);
        auto va_i32_hi = _mm_sub_epi32(MlasShiftRightInt32<XInt8>(_mm_unpackhi_epi16(va_i16x8, va_i16x8), 24), ZeroPointVectorIn);
        va_i32_lo = MlasQuantizeLinearVector(_mm_cvtepi32_ps(va_i32_lo), ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVectorOut);
        va_i32_hi = MlasQuantizeLinearVector(_mm_cvtepi32_ps(va_i32_hi), ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVectorOut);
        va_i32_lo = MlasQuantizeLinearPackBytes<XInt8>(va_i32_lo);
        va_i32_hi = MlasQuantizeLinearPackBytes<XInt8>(va_i32_hi);

        if (N >= 4) {
            *((int32_t*)Output) = _mm_cvtsi128_si32(va_i32_lo);
            va_i32_lo = va_i32_hi;
            Output += 4;
            N -= 4;
        }

        uint32_t PackedValueC = (uint32_t)_mm_cvtsi128_si32(va_i32_lo);
        for (size_t i = 0; i < N; ++i) {
            *((uint8_t*)Output + i) = (uint8_t)PackedValueC;
            PackedValueC >>= 8;
        }
    }
}

#elif defined(MLAS_NEON_INTRINSICS)

template <typename XInt8>
void
MLASCALL
MlasRequantizeLinear(
    const XInt8* Input,
    float ScaleIn,
    int32_t ZeroPointIn,
    XInt8* Output,
    float ScaleOut,
    int32_t ZeroPointOut,
    size_t N
    )
{
    typedef MLAS_SignedUnsignedIntOps<XInt8> SUI;

    auto ScaleVector = MlasBroadcastFloat32x4(ScaleOut / ScaleIn);
    auto MinimumValueVector = MlasBroadcastFloat32x4(float((int)std::numeric_limits<XInt8>::min() - ZeroPointOut));
    auto MaximumValueVector = MlasBroadcastFloat32x4(float((int)std::numeric_limits<XInt8>::max() - ZeroPointOut));
    auto ZeroPointVectorOut = MlasBroadcastInt32x4(ZeroPointOut);
    const typename SUI::i8x8_t ZeroPointVectorIn = SUI::vmov_n_i8((XInt8)ZeroPointIn);

    for(;N >= 8; N -= 8) {
        const typename SUI::i8x8_t va_i8x8 = SUI::vld1_i8(Input);
        Input += 8;

        const int16x8_t va_i16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(va_i8x8, ZeroPointVectorIn));
        auto va_i32_lo = vmovl_s16(vget_low_s16(va_i16x8));
        auto va_i32_hi = vmovl_s16(vget_high_s16(va_i16x8));
        va_i32_lo = MlasQuantizeLinearVector(vcvtq_f32_s32(va_i32_lo), ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVectorOut);
        va_i32_hi = MlasQuantizeLinearVector(vcvtq_f32_s32(va_i32_hi), ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVectorOut);
        va_i32_lo = MlasQuantizeLinearPackBytes<XInt8>(va_i32_lo);
        va_i32_hi = MlasQuantizeLinearPackBytes<XInt8>(va_i32_hi);

        *((int32_t*)Output) = vgetq_lane_s32(va_i32_lo, 0);
        *((int32_t*)(Output + 4)) = vgetq_lane_s32(va_i32_hi, 0);

        Output += 8;
    }

    if (N > 0) {
        XInt8 tail[8];
        memcpy(tail, Input, N);

        const typename SUI::i8x8_t va_i8x8 = SUI::vld1_i8(tail);
        Input += 8;

        const int16x8_t va_i16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(va_i8x8, ZeroPointVectorIn));
        auto va_i32_lo = vmovl_s16(vget_low_s16(va_i16x8));
        auto va_i32_hi = vmovl_s16(vget_high_s16(va_i16x8));
        va_i32_lo = MlasQuantizeLinearVector(vcvtq_f32_s32(va_i32_lo), ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVectorOut);
        va_i32_hi = MlasQuantizeLinearVector(vcvtq_f32_s32(va_i32_hi), ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVectorOut);
        va_i32_lo = MlasQuantizeLinearPackBytes<XInt8>(va_i32_lo);
        va_i32_hi = MlasQuantizeLinearPackBytes<XInt8>(va_i32_hi);

        if (N >= 4) {
            *(uint32_t*)Output = vgetq_lane_s32(va_i32_lo, 0);
            va_i32_lo = va_i32_hi;
            Output += 4;
            N -= 4;
        }

        uint32_t PackedValueC = (uint32_t)vgetq_lane_s32(va_i32_lo, 0);
        for (size_t i = 0; i < N; ++i) {
            *((uint8_t*)Output + i) = (uint8_t)PackedValueC;
            PackedValueC >>= 8;
        }
    }
}

#else

template <typename XInt8>
void
MLASCALL
MlasRequantizeLinear(
    const XInt8* Input,
    float ScaleIn,
    int32_t ZeroPointIn,
    XInt8* Output,
    float ScaleOut,
    int32_t ZeroPointOut,
    size_t N
    )
{
    const float MinimumValue = (float)((int)std::numeric_limits<XInt8>::min() - ZeroPointOut);
    const float MaximumValue = (float)((int)std::numeric_limits<XInt8>::max() - ZeroPointOut);

    float scale = ScaleIn / ScaleOut;
    for (; N > 0; N--) {
        int32_t input_i32 = static_cast<int32_t>(*Input++) - ZeroPointIn;
        float deq_value = input_i32 * scale;
        deq_value = std::min(std::max(deq_value, MinimumValue), MaximumValue);
        *Output++ = (XInt8)(int32_t)std::nearbyintf(deq_value + ZeroPointOut);
    }
}

#endif


template
void
MLASCALL
MlasRequantizeLinear<uint8_t>(
    const uint8_t* Input,
    float ScaleIn,
    int32_t ZeroPointIn,
    uint8_t* Output,
    float ScaleOut,
    int32_t ZeroPointOut,
    size_t N
    );


template
void
MLASCALL
MlasRequantizeLinear<int8_t>(
    const int8_t* Input,
    float ScaleIn,
    int32_t ZeroPointIn,
    int8_t* Output,
    float ScaleOut,
    int32_t ZeroPointOut,
    size_t N
    );
