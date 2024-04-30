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

#include "KernelGenerator.h"

#include "ops/BinaryArithmeticLayer.h"
#include "ops/ConvolutionLayer.h"
#include "ops/DepthwiseConvolutionLayer.h"
#include "ops/ElementwiseActivationLayer.h"
#include "ops/FullyConnectedLayer.h"
#include "ops/LossMeanSquaredErrorLayer.h"
#include "ops/LossCategoricalCrossentropyLayer.h"
#include "ops/MeanLayer.h"
#include "ops/GradientApplier.h"
#include "ops/PadLayer.h"
#include "ops/PoolLayer.h"
#include "ops/ReshapeLayer.h"
#include "ops/SoftMaxLayer.h"

#include <backend/Backend.h>
#include <backend/IConfig.h>
#include <memory>
#include <util/Utils.h>
#include <util/logging.h>
#include <exec/DynamicShapeInferer.h>

#include <stdexcept>

namespace onert
{
namespace backend
{
namespace train
{

namespace
{
ops::ElementwiseActivationType
convertElementwiseActivationType(ir::operation::ElementwiseActivation::Type type_ir)
{
  switch (type_ir)
  {
    case ir::operation::ElementwiseActivation::Type::RELU:
      return ops::ElementwiseActivationType::kReLU;
    default:
      throw std::runtime_error("train KernelGenerator : Not supported operation yet");
  }
}

ops::PoolType convertPoolType(ir::operation::Pool2D::PoolType type_ir)
{
  switch (type_ir)
  {
    // TODO Implement AVG PoolType
    case ir::operation::Pool2D::PoolType::MAX:
      return ops::PoolType::kMax;
    default:
      throw std::runtime_error("train KernelGenerator : Not supported operation yet");
  }
}

std::unique_ptr<ops::GradientApplier>
generateGradientApplier(const exec::train::optimizer::Optimizer *optimizer,
                        const IPortableTensor *gradient, ITrainableTensor *trainable)
{
  auto update_fn = std::make_unique<ops::GradientApplier>();
  update_fn->configure(optimizer, gradient, trainable);
  return update_fn;
}
} // namespace

std::unique_ptr<exec::train::TrainableFnSequence> KernelGenerator::generate(ir::OperationIndex idx)
{
  auto ret = std::make_unique<exec::train::TrainableFnSequence>();

  const auto &op = _tgraph.operation(idx);
  op.accept(*this);
  assert(_return_fn);
  ret->append(std::move(_return_fn));

  for (auto &&update_fn : _update_funcs)
    ret->append(std::move(update_fn));
  _update_funcs.clear();

  for (auto &&ind : (op.getInputs() | ir::Remove::UNDEFINED) + op.getOutputs())
  {
    auto portable_tensor = _tensor_reg->getPortableTensor(ind);
    if (portable_tensor)
    {
      assert(portable_tensor->layout() == ir::Layout::NHWC);
    }
    auto tensor = _tensor_reg->getNonConstTensor(ind);
    if (tensor)
    {
      tensor->increase_ref();
    }
  }
  return ret;
}

KernelGenerator::KernelGenerator(const ir::train::TrainableGraph &tgraph,
                                 const std::shared_ptr<TensorRegistry> &tensor_reg,
                                 const std::shared_ptr<ExternalContext> &external_context,
                                 const exec::train::optimizer::Optimizer *optimizer)
  : backend::train::KernelGeneratorBase{tgraph}, _current_layout{tgraph.layout()},
    _tensor_reg{tensor_reg}, _external_context(external_context), _optimizer{optimizer},
    _update_funcs{}
{
  // DO NOTHING
}

void KernelGenerator::visit(const ir::train::operation::BinaryArithmetic &node)
{
  using ir::train::operation::BinaryArithmetic;

  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(BinaryArithmetic::Input::LHS)};
  const auto rhs_index{node.getInputs().at(BinaryArithmetic::Input::RHS)};

  const auto arithmetic_type = node.param().arithmetic_type;
  const auto activation = node.param().activation;

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto lhs_tensor = _tensor_reg->getPortableTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getPortableTensor(rhs_index);

  auto fn = std::make_unique<ops::BinaryArithmeticLayer>();
  fn->configure(lhs_tensor, rhs_tensor, output_tensor, activation,
                static_cast<cpu::ops::ArithmeticType>(arithmetic_type));

  if (node.isRequiredForBackward())
  {
    auto back_prop_output_tensor = _tensor_reg->getBackPropTensor(output_index);
    auto back_prop_lhs_tensor = _tensor_reg->getBackPropTensor(lhs_index);
    auto back_prop_rhs_tensor = _tensor_reg->getBackPropTensor(rhs_index);

    fn->configureBackward(back_prop_lhs_tensor, back_prop_rhs_tensor, back_prop_output_tensor,
                          activation, static_cast<train::ops::ArithmeticType>(arithmetic_type));
  }
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::Conv2D &node)
{
  using ir::train::operation::Conv2D;

  const auto out_index{node.getOutputs().at(0)};
  const auto in_index{node.getInputs().at(Conv2D::Input::INPUT)};
  const auto ker_index{node.getInputs().at(Conv2D::Input::KERNEL)};
  const auto bias_index{node.getInputs().at(Conv2D::Input::BIAS)};

  auto out_tensor = _tensor_reg->getPortableTensor(out_index);
  auto in_tensor = _tensor_reg->getPortableTensor(in_index);
  auto ker_tensor = _tensor_reg->getTrainableTensor(ker_index);
  auto bias_tensor = _tensor_reg->getTrainableTensor(bias_index);

  // Generate kernel
  const auto stride = node.param().stride;
  const auto activation = node.param().activation;
  const auto &param_padding = node.param().padding;
  const auto dilation = node.param().dilation;
  auto fn = std::make_unique<ops::ConvolutionLayer>();

  auto &operands = _tgraph.operands();
  const auto ifm_shape = operands.at(in_index).shape().asFeature(_current_layout);
  const auto ofm_shape = operands.at(out_index).shape().asFeature(_current_layout);
  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in].
  const auto &ker_shape = operands.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);

  const auto padding =
    ir::calculatePadding(param_padding, ifm_shape, ofm_shape, stride, ker_width, ker_height,
                         dilation.width_factor, dilation.height_factor);

  const bool is_cacheable_weights = false;
  fn->configure(in_tensor, ker_tensor, bias_tensor, param_padding.type, padding.left, padding.right,
                padding.top, padding.bottom, stride.horizontal, stride.vertical,
                dilation.width_factor, dilation.height_factor, activation, out_tensor,
                is_cacheable_weights);

  auto ker_grad_tensor = _tensor_reg->getGradientTensor(ker_index);
  auto bias_grad_tensor = _tensor_reg->getGradientTensor(bias_index);

  if (node.isRequiredForBackward())
  {

    auto out_back_prop_tensor = _tensor_reg->getBackPropTensor(out_index);
    auto in_back_prop_tensor = _tensor_reg->getBackPropTensor(in_index);

    fn->configureBackward(ker_tensor, in_back_prop_tensor, ker_grad_tensor, bias_grad_tensor,
                          out_back_prop_tensor, activation);

    // Generate GradientApplier
    if (bias_tensor)
      _update_funcs.emplace_back(
        generateGradientApplier(_optimizer, bias_grad_tensor, bias_tensor));
    _update_funcs.emplace_back(generateGradientApplier(_optimizer, ker_grad_tensor, ker_tensor));
  }

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::DepthwiseConv2D &node)
{
  using ir::train::operation::DepthwiseConv2D;

  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(DepthwiseConv2D::Input::INPUT)};
  const auto ker_index{node.getInputs().at(DepthwiseConv2D::Input::KERNEL)};
  const auto bias_index{node.getInputs().at(DepthwiseConv2D::Input::BIAS)};

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);
  auto ker_tensor = _tensor_reg->getTrainableTensor(ker_index);
  auto bias_tensor = _tensor_reg->getTrainableTensor(bias_index);

  const auto stride = node.param().stride;
  const auto &operands = _tgraph.operands();
  const auto ofm_shape = operands.at(ofm_index).shape().asFeature(_current_layout);
  const auto ifm_shape = operands.at(ifm_index).shape().asFeature(_current_layout);
  // Kernel format is [1, kernel_height, kernel_width, depth_out].
  const auto &ker_shape = operands.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);
  const auto dilation_width = node.param().dilation.width_factor;
  const auto dilation_height = node.param().dilation.height_factor;
  const auto padding = ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride,
                                            ker_width, ker_height, dilation_width, dilation_height);
  const auto multiplier = node.param().multiplier;
  const auto activation = node.param().activation;

  auto fn = std::make_unique<ops::DepthwiseConvolutionLayer>();

  fn->configure(ifm_tensor, ker_tensor, bias_tensor, padding.left, padding.right, padding.top,
                padding.bottom, stride.horizontal, stride.vertical, multiplier, dilation_width,
                dilation_height, activation, ofm_tensor, _external_context);

  auto ker_grad_tensor = _tensor_reg->getGradientTensor(ker_index);
  auto bias_grad_tensor = _tensor_reg->getGradientTensor(bias_index);

  if (node.isRequiredForBackward())
  {
    auto ofm_back_prop_tensor = _tensor_reg->getBackPropTensor(ofm_index);
    auto ifm_back_prop_tensor = _tensor_reg->getBackPropTensor(ifm_index);

    fn->configureBackward(ifm_back_prop_tensor, ker_grad_tensor, bias_grad_tensor,
                          ofm_back_prop_tensor, activation);

    // Generate GradientApplier
    if (bias_tensor)
      _update_funcs.emplace_back(
        generateGradientApplier(_optimizer, bias_grad_tensor, bias_tensor));
    _update_funcs.emplace_back(generateGradientApplier(_optimizer, ker_grad_tensor, ker_tensor));
  }

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::ElementwiseActivation &node)
{
  using ir::train::operation::ElementwiseActivation;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ElementwiseActivation::Input::INPUT)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto fn = std::make_unique<ops::ElementwiseActivationLayer>();

  auto convertToInferActivationType = [](const ir::operation::ElementwiseActivation::Type &type) {
    switch (type)
    {
      case ir::operation::ElementwiseActivation::Type::RELU:
        return cpu::ops::ElementwiseActivationType::kReLU;
      default:
        throw std::invalid_argument("Unsupported ElementwiseActivation::Type");
    }
  };

  fn->configure(input_tensor, output_tensor, node.param().alpha, node.param().beta,
                convertToInferActivationType(node.param().op_type));

  if (node.isRequiredForBackward())
  {
    auto back_prop_input_tensor = _tensor_reg->getBackPropTensor(input_index);
    auto back_prop_output_tensor = _tensor_reg->getBackPropTensor(output_index);

    fn->configureBackward(input_tensor, back_prop_input_tensor, back_prop_output_tensor,
                          node.param().alpha, node.param().beta,
                          convertElementwiseActivationType(node.param().op_type));
  }

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::FullyConnected &node)
{
  using ir::train::operation::FullyConnected;

  const auto out_index{node.getOutputs().at(0)};
  const auto in_index{node.getInputs().at(FullyConnected::Input::INPUT)};
  const auto weights_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};
  const auto bias_index{node.getInputs().at(FullyConnected::Input::BIAS)};

  auto out_tensor = _tensor_reg->getPortableTensor(out_index);
  auto in_tensor = _tensor_reg->getPortableTensor(in_index);
  auto weights_tensor = _tensor_reg->getTrainableTensor(weights_index);
  auto bias_tensor = _tensor_reg->getTrainableTensor(bias_index);

  // Generate kernel
  const auto activation = node.param().activation;
  const auto weights_format = node.param().weights_format;

  auto fn = std::make_unique<ops::FullyConnectedLayer>();

  fn->configure(in_tensor, weights_tensor, bias_tensor, activation, weights_format, out_tensor,
                _external_context);

  if (node.isRequiredForBackward())
  {
    auto out_back_prop_tensor = _tensor_reg->getBackPropTensor(out_index);
    auto in_back_prop_tensor = _tensor_reg->getBackPropTensor(in_index);
    auto weights_grad_tensor = _tensor_reg->getGradientTensor(weights_index);
    auto bias_grad_tensor = _tensor_reg->getGradientTensor(bias_index);

    fn->configureBackward(in_tensor, weights_tensor, out_tensor, in_back_prop_tensor,
                          weights_grad_tensor, bias_grad_tensor, out_back_prop_tensor, activation,
                          weights_format);

    // Generate GradientAppliers
    if (bias_tensor)
      _update_funcs.emplace_back(
        generateGradientApplier(_optimizer, bias_grad_tensor, bias_tensor));
    _update_funcs.emplace_back(
      generateGradientApplier(_optimizer, weights_grad_tensor, weights_tensor));
  }

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::Loss &node)
{
  using ir::train::operation::Loss;

  const auto output_index{node.getOutputs().at(0)};
  const auto y_pred_index{node.getInputs().at(Loss::Y_PRED)};
  const auto y_true_index{node.getInputs().at(Loss::Y_TRUE)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto y_pred_tensor = _tensor_reg->getPortableTensor(y_pred_index);
  auto y_true_tensor = _tensor_reg->getPortableTensor(y_true_index);

  auto back_prop_y_pred_tensor = _tensor_reg->getBackPropTensor(y_pred_index);

  auto loss_code = node.param().loss_code;
  auto loss_param = node.param().loss_param;

  switch (loss_code)
  {
    case ir::train::LossCode::MeanSquaredError:
    {
      auto fn = std::make_unique<ops::LossMeanSquaredErrorLayer>();
      fn->configure(y_pred_tensor, y_true_tensor, output_tensor, back_prop_y_pred_tensor);
      _return_fn = std::move(fn);
      break;
    }
    case ir::train::LossCode::CategoricalCrossentropy:
    {
      auto fn = std::make_unique<ops::LossCategoricalCrossentropyLayer>();
      fn->configure(y_pred_tensor, y_true_tensor, output_tensor, back_prop_y_pred_tensor,
                    loss_param.cce.axis, loss_param.cce.label_smoothing);
      _return_fn = std::move(fn);
      break;
    }
    default:
      throw std::runtime_error("LossLayer: unsupported loss type");
      break;
  }
}

void KernelGenerator::visit(const ir::train::operation::Pad &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Pad::Input::INPUT)};
  const auto pad_index{node.getInputs().at(ir::operation::Pad::Input::PAD)};
  const auto output_index{node.getOutputs().at(0)};

  auto input = _tensor_reg->getPortableTensor(input_index);
  auto pad = _tensor_reg->getPortableTensor(pad_index);
  auto output = _tensor_reg->getPortableTensor(output_index);

  auto fn = std::make_unique<ops::PadLayer>();

  IPortableTensor *value = nullptr;
  if (node.getInputs().size() == 3) // isPadV2
  {
    const auto value_index{node.getInputs().at(ir::operation::Pad::Input::VALUE)};
    value = _tensor_reg->getPortableTensor(value_index);
  }

  fn->configure(input, pad, value, output);

  if (node.isRequiredForBackward())
  {
    auto out_back_prop_tensor = _tensor_reg->getBackPropTensor(output_index);
    auto in_back_prop_tensor = _tensor_reg->getBackPropTensor(input_index);
    fn->configureBackward(in_back_prop_tensor, out_back_prop_tensor);
  }
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::Pool2D &node)
{
  using ir::train::operation::Pool2D;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  const auto &operands = _tgraph.operands();
  const auto &ofm_shape = operands.at(output_index).shape();
  const auto &ifm_shape = operands.at(input_index).shape();

  if (ifm_shape.rank() != 4)
  {
    std::runtime_error(node.name() + " only supports 4D tensor as input");
  }

  // calcualate padding
  const auto stride = node.param().stride;
  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto padding =
    ir::calculatePadding(node.param().padding, ifm_shape.asFeature(_current_layout),
                         ofm_shape.asFeature(_current_layout), stride, kw, kh);

  auto out_tensor = _tensor_reg->getPortableTensor(output_index);
  auto in_tensor = _tensor_reg->getPortableTensor(input_index);

  const auto activation = node.param().activation;
  const auto pool_type = convertPoolType(node.param().op_type);

  auto fn = std::make_unique<ops::PoolLayer>();

  auto convertToInferPoolType = [](const train::ops::PoolType &pool_type) {
    switch (pool_type)
    {
      case train::ops::PoolType::kMax:
        return cpu::ops::PoolType::kMax;
      default:
        throw std::runtime_error("PoolLayer: Unsupported pool type yet");
    }
  };

  fn->configure(in_tensor, padding.left, padding.right, padding.top, padding.bottom,
                stride.horizontal, stride.vertical, kw, kh, activation, out_tensor,
                convertToInferPoolType(pool_type));

  if (node.isRequiredForBackward())
  {
    auto out_back_prop_tensor = _tensor_reg->getBackPropTensor(output_index);
    auto in_back_prop_tensor = _tensor_reg->getBackPropTensor(input_index);
    fn->configureBackward(padding.left, padding.right, padding.top, padding.bottom,
                          stride.horizontal, stride.vertical, kw, kh, activation, pool_type,
                          out_tensor, in_back_prop_tensor, out_back_prop_tensor);
  }

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::Reduce &node)
{
  using ir::train::operation::Reduce;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(Reduce::Input::INPUT)};
  const auto axes_index{node.getInputs().at(Reduce::Input::AXES)};

  const auto keep_dims = node.param().keep_dims;

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto axes_tensor = _tensor_reg->getPortableTensor(axes_index);

  if (node.param().reduce_type == ir::operation::Reduce::ReduceType::MEAN)
  {
    auto fn = std::make_unique<ops::MeanLayer>();
    fn->configure(input_tensor, axes_tensor, output_tensor, keep_dims);
    if (node.isRequiredForBackward())
    {
      auto back_prop_output_tensor = _tensor_reg->getBackPropTensor(output_index);
      auto back_prop_input_tensor = _tensor_reg->getBackPropTensor(input_index);
      fn->configureBackward(back_prop_input_tensor, back_prop_output_tensor);
    }
    _return_fn = std::move(fn);
  }
  else
  {
    throw std::runtime_error("ReduceLayer: unsupported reduce type");
  }
}

void KernelGenerator::visit(const ir::train::operation::Reshape &node)
{
  using ir::train::operation::Reshape;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reshape::Input::INPUT)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  // optional 2nd input
  IPortableTensor *shape_tensor = nullptr;

  if (node.getInputs().size() == 2)
  {
    const auto shape_index{node.getInputs().at(ir::operation::Reshape::Input::SHAPE)};
    shape_tensor = _tensor_reg->getPortableTensor(shape_index);
  }

  auto fn = std::make_unique<ops::ReshapeLayer>();

  fn->configure(input_tensor, shape_tensor, output_tensor);
  if (node.isRequiredForBackward())
  {
    auto input_back_prop_tensor = _tensor_reg->getBackPropTensor(input_index);
    auto output_back_prop_tensor = _tensor_reg->getBackPropTensor(output_index);
    fn->configureBackward(input_back_prop_tensor, output_back_prop_tensor);
  }
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::Softmax &node)
{
  using ir::train::operation::Softmax;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Softmax::Input::INPUT)};

  const auto beta = node.param().beta;

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto fn = std::make_unique<ops::SoftMaxLayer>();

  fn->configure(input_tensor, beta, output_tensor);

  if (node.isRequiredForBackward())
  {
    auto input_back_prop_tensor = _tensor_reg->getBackPropTensor(input_index);
    auto output_back_prop_tensor = _tensor_reg->getBackPropTensor(output_index);
    fn->configureBackward(input_back_prop_tensor, output_back_prop_tensor);
  }
  _return_fn = std::move(fn);
}

} // namespace train
} // namespace backend
} // namespace onert
