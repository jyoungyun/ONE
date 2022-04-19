/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "GenModelTest.h"

struct LessEqualVariationParam
{
  TestCaseData tcd;
  circle::TensorType input_type = circle::TensorType::TensorType_FLOAT32;
  const std::vector<std::string> backends = {"acl_cl", "acl_neon", "cpu"};
};

class LessEqualVariation : public GenModelTest,
                           public ::testing::WithParamInterface<LessEqualVariationParam>
{
};

// Input shape:
//   Base: {1, 2, 2, 1}
//   Brodcast: {1} on of two input
// Output shape: {1, 2, 2, 1}
// Input type: Non-quantization type
// Output type: BOOL
// Test with different input type and value
INSTANTIATE_TEST_SUITE_P(GenModelTest, LessEqualVariation,
                         ::testing::Values(
                           // Float type
                           LessEqualVariationParam{TestCaseData{}
                                                     .addInput<float>({0.1, 0.3, 0.2, 0.7})
                                                     .addInput<float>({0.1, 0.2, 0.3, 0.4})
                                                     .addOutput<bool>({true, false, true, false})},
                           // Float type - broadcast
                           LessEqualVariationParam{TestCaseData{}
                                                     .addInput<float>({0.1, 0.3, 0.2, 0.7})
                                                     .addInput<float>({0.3})
                                                     .addOutput<bool>({true, true, true, false})},
                           // Int32 type
                           LessEqualVariationParam{TestCaseData{}
                                                     .addInput<int32_t>({1, 3, 2, 7})
                                                     .addInput<int32_t>({1, 2, 3, 4})
                                                     .addOutput<bool>({true, false, true, false}),
                                                   circle::TensorType::TensorType_INT32},
                           // Int32 type - broadcast
                           LessEqualVariationParam{TestCaseData{}
                                                     .addInput<int32_t>({1, 3, 2, 7})
                                                     .addInput<int32_t>({5})
                                                     .addOutput<bool>({true, true, true, false}),
                                                   circle::TensorType::TensorType_INT32},
                           // Int64 type
                           // NYI: acl backend
                           LessEqualVariationParam{TestCaseData{}
                                                     .addInput<int64_t>({1, 3, -2, 7})
                                                     .addInput<int64_t>({1, 2, 3, 4})
                                                     .addOutput<bool>({true, false, true, false}),
                                                   circle::TensorType::TensorType_INT64,
                                                   {"cpu"}},
                           // Int64 type - broadcast
                           // NYI: acl backend
                           LessEqualVariationParam{TestCaseData{}
                                                     .addInput<int64_t>({1, 3, -2, 7})
                                                     .addInput<int64_t>({1})
                                                     .addOutput<bool>({true, false, true, false}),
                                                   circle::TensorType::TensorType_INT64,
                                                   {"cpu"}}));

TEST_P(LessEqualVariation, Test)
{
  auto &param = GetParam();

  auto lhs_data = param.tcd.inputs.at(0);
  auto rhs_data = param.tcd.inputs.at(1);

  bool broadcast_lhs = false;
  bool broadcast_rhs = false;
  if (lhs_data.size() != rhs_data.size())
  {
    if (lhs_data.size() < rhs_data.size())
      broadcast_lhs = true;
    else
      broadcast_rhs = true;
  }

  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_BOOL;

  int lhs = broadcast_lhs ? cgen.addTensor({{1}, param.input_type})
                          : cgen.addTensor({{1, 2, 2, 1}, param.input_type});
  int rhs = broadcast_rhs ? cgen.addTensor({{1}, param.input_type})
                          : cgen.addTensor({{1, 2, 2, 1}, param.input_type});
  int out = cgen.addTensor({{1, 2, 2, 1}, output_type});
  cgen.addOperatorLessEqual({{lhs, rhs}, {out}});
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(param.tcd);
  _context->setBackends(param.backends);

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_LessEqual_DifferentType)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_BOOL});
  cgen.addOperatorLessEqual({{lhs, rhs}, {out}});
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_LessEqual_InvalidType)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorLessEqual({{lhs, rhs}, {out}});
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
