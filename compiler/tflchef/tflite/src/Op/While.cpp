/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "While.h"

#include "Convert.h"
#include "FillerHelper.h"

namespace tflchef
{

void TFliteOpWhile::filler(const tflite::Operator *op, TFliteImport *import,
                           tflchef::ModelRecipe *model_recipe) const
{
  const auto &inputs = *op->inputs();

  for (int input : inputs)
  {
    fill_tensor_to_import(input, import);
  }
}

tflchef::Operation *TFliteOpWhile::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_WhileOptions();
  assert(op_params != nullptr);

  operation->set_type("While");

  auto op_options = operation->mutable_while_options();

  op_options->set_body_subgraph_index(op_params->body_subgraph_index());
  op_options->set_cond_subgraph_index(op_params->cond_subgraph_index());

  return operation;
}

} // namespace tflchef
