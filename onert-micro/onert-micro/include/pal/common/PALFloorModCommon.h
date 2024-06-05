/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ONERT_MICRO_EXECUTE_PAL_FLOORMOD_COMMON_H
#define ONERT_MICRO_EXECUTE_PAL_FLOORMOD_COMMON_H

#include "OMStatus.h"
#include "core/OMRuntimeShape.h"
#include "PALBinaryOpCommon.h"

namespace onert_micro
{
namespace execute
{
namespace pal
{

inline void FloorMod(const int flat_size, const float *input1_data, const float *input2_data,
                     float *output_data)
{
  BinaryOp<float, FloorModFn<float>>(flat_size, input1_data, input2_data, output_data);
}

inline void

BroadcastFloorMod4DSlow(const core::OMRuntimeShape &input1_shape, const float *input1_data,
                        const core::OMRuntimeShape &input2_shape, const float *input2_data,
                        const core::OMRuntimeShape &output_shape, float *output_data)
{
  BroadcastBinaryOp4DSlow<float, FloorModFn<float>>(input1_shape, input1_data, input2_shape,
                                                    input2_data, output_shape, output_data);
}

} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_PAL_FLOORMOD_COMMON_H
