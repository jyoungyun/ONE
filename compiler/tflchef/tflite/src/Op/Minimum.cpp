/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Minimum.h"

#include "Convert.h"
#include "FillerHelper.h"

namespace tflchef
{

void TFliteOpMinimum::filler(const tflite::Operator *op, TFliteImport *import,
                             tflchef::ModelRecipe *model_recipe) const
{
  fill_two_inputs(op, import);
}

tflchef::Operation *TFliteOpMinimum::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  operation->set_type("Minimum");

  return operation;
}

} // namespace tflchef
