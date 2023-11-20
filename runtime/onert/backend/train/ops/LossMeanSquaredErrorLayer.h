/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_TRAIN_OPS_LOSS_MEANSQUAREDERROR_LAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_LOSS_MEANSQUAREDERROR_LAYER_H__

#include "LossLayer.h"

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

class LossMeanSquaredErrorLayer : public LossLayer
{
public:
  LossMeanSquaredErrorLayer() = default;

  void configure(const IPortableTensor *y_pred, const IPortableTensor *y_true,
                 IPortableTensor *output, IPortableTensor *back_prop_y_pred,
                 LossReductionType reduction_type);
  void forward(bool training) override;
  void backward() override;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_LOSS_MEANSQUAREDERROR_LAYER_H__
