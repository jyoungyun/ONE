/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef ONERT_MICRO_EXECUTE_PAL_ADD_H
#define ONERT_MICRO_EXECUTE_PAL_ADD_H

#include "PALAddCommon.h"

namespace onert_micro
{
namespace execute
{
namespace pal
{
template <>
OMStatus Add<int8_t>(const core::BinaryArithmeticBroadcastParams &params, const int flat_size,
                     const int8_t *input1_data, const int8_t *input2_data, int8_t *output_data)
{
  assert(false && "Not IMPL yet");
}

template <>
OMStatus Add<int16_t>(const core::BinaryArithmeticBroadcastParams &params, const int flat_size,
                      const int16_t *input1_data, const int16_t *input2_data, int16_t *output_data)
{
  assert(false && "Not IMPL yet");
}

} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_PAL_ADD_H
