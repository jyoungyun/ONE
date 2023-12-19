/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "allocation.h"
#include "args.h"
#include "benchmark.h"
#include "measure.h"
#include "nnfw.h"
#include "nnfw_util.h"
#include "nnfw_internal.h"
#include "nnfw_experimental.h"
#include "randomgen.h"
#include "rawformatter.h"
#include "rawdataloader.h"

#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <libgen.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

static const char *default_backend_cand = "train";

int main(const int argc, char **argv)
{
  using namespace onert_train;

  try
  {
    Args args(argc, argv);
    if (args.printVersion())
    {
      uint32_t version;
      NNPR_ENSURE_STATUS(nnfw_query_info_u32(NULL, NNFW_INFO_ID_VERSION, &version));
      std::cout << "onert_train (nnfw runtime: v" << (version >> 24) << "."
                << ((version & 0x0000FF00) >> 8) << "." << (version & 0xFF) << ")" << std::endl;
      exit(0);
    }

    // TODO Apply verbose level to phases
    const int verbose = args.getVerboseLevel();

    // prepare measure tool
    Measure measure(args.getMemoryPoll());

    nnfw_session *session = nullptr;
    NNPR_ENSURE_STATUS(nnfw_create_session(&session));

    // ModelLoad
    measure.run(PhaseType::MODEL_LOAD, [&]() {
      if (args.useSingleModel())
        NNPR_ENSURE_STATUS(
          nnfw_load_model_from_modelfile(session, args.getModelFilename().c_str()));
      else
        NNPR_ENSURE_STATUS(nnfw_load_model_from_file(session, args.getPackageFilename().c_str()));
    });

    // Set training backend
    NNPR_ENSURE_STATUS(nnfw_set_available_backends(session, default_backend_cand));

    uint32_t num_inputs;
    NNPR_ENSURE_STATUS(nnfw_input_size(session, &num_inputs));

    uint32_t num_expecteds;
    NNPR_ENSURE_STATUS(nnfw_output_size(session, &num_expecteds));

    // verify input and output

    auto verifyInputTypes = [session]() {
      uint32_t sz;
      NNPR_ENSURE_STATUS(nnfw_input_size(session, &sz));
      for (uint32_t i = 0; i < sz; ++i)
      {
        nnfw_tensorinfo ti;
        NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));

        if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
        {
          std::cerr << "E: not supported input type" << std::endl;
          exit(-1);
        }
      }
    };

    auto verifyOutputTypes = [session]() {
      uint32_t sz;
      NNPR_ENSURE_STATUS(nnfw_output_size(session, &sz));

      for (uint32_t i = 0; i < sz; ++i)
      {
        nnfw_tensorinfo ti;
        NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));

        if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
        {
          std::cerr << "E: not supported output type" << std::endl;
          exit(-1);
        }
      }
    };

    verifyInputTypes();
    verifyOutputTypes();

    auto convertLossType = [](int type) {
      switch (type)
      {
        case 0:
          return NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR;
        case 1:
          return NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY;
        default:
          std::cerr << "E: not supported loss type" << std::endl;
          exit(-1);
      }
    };

    auto convertLossReductionType = [](int type) {
      switch (type)
      {
        case 0:
          return NNFW_TRAIN_LOSS_REDUCTION_INVALID;
        case 1:
          return NNFW_TRAIN_LOSS_REDUCTION_SUM_OVER_BATCH_SIZE;
        case 2:
          return NNFW_TRAIN_LOSS_REDUCTION_SUM;
        default:
          std::cerr << "E: not supported loss reduction type" << std::endl;
          exit(-1);
      }
    };

    auto convertOptType = [](int type) {
      switch (type)
      {
        case 0:
          return NNFW_TRAIN_OPTIMIZER_SGD;
        case 1:
          return NNFW_TRAIN_OPTIMIZER_ADAM;
        default:
          std::cerr << "E: not supported optimizer type" << std::endl;
          exit(-1);
      }
    };

    // prepare training info
    nnfw_train_info tri;
    tri.batch_size = args.getBatchSize();
    tri.learning_rate = args.getLearningRate();
    tri.loss_info.loss = convertLossType(args.getLossType());
    tri.loss_info.reduction_type = convertLossReductionType(args.getLossReductionType());
    tri.opt = convertOptType(args.getOptimizerType());

    std::cout << "== training parameter ==" << std::endl;
    std::cout << tri;
    std::cout << "========================" << std::endl;
    // prepare execution

    // TODO When nnfw_{prepare|run} are failed, can't catch the time
    measure.run(PhaseType::PREPARE,
                [&]() { NNPR_ENSURE_STATUS(nnfw_train_prepare(session, &tri)); });

    // prepare input and expected tensor info lists
    std::vector<nnfw_tensorinfo> input_infos;
    std::vector<nnfw_tensorinfo> expected_infos;

    // prepare data buffers
    std::vector<Allocation> input_data(num_inputs);
    std::vector<Allocation> expected_data(num_expecteds);

    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));
      input_data[i].alloc(bufsize_for(&ti));
      input_infos.emplace_back(std::move(ti));
    }

    for (uint32_t i = 0; i < num_expecteds; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));
      expected_data[i].alloc(bufsize_for(&ti));
      expected_infos.emplace_back(std::move(ti));
    }

    uint32_t data_length;

    Generator generator;
    RawDataLoader rawDataLoader;

    if (!args.getLoadRawInputFilename().empty() && !args.getLoadRawExpectedFilename().empty())
    {
      std::tie(generator, data_length) =
        rawDataLoader.loadData(args.getLoadRawInputFilename(), args.getLoadRawExpectedFilename(),
                               input_infos, expected_infos, tri.batch_size);
      // TODO get data generator for validation
    }
    else
    {
      // TODO Use random generator
      std::cerr << "E: not supported random input and expected generator" << std::endl;
      exit(-1);
    }

    std::vector<float> losses(num_expecteds);
    measure.run(PhaseType::EXECUTE, [&]() {
      const int num_step = data_length / tri.batch_size;
      const int num_epoch = args.getEpoch();
      measure.set(num_epoch, num_step);
      for (uint32_t epoch = 0; epoch < num_epoch; ++epoch)
      {
        //
        // TRAINING
        //
        {
          std::fill(losses.begin(), losses.end(), 0);
          for (uint32_t n = 0; n < num_step; ++n)
          {
            // get batchsize data
            if (!generator(n, input_data, expected_data))
              break;

            // prepare input
            for (uint32_t i = 0; i < num_inputs; ++i)
            {
              NNPR_ENSURE_STATUS(
                nnfw_train_set_input(session, i, input_data[i].data(), &input_infos[i]));
            }

            // prepare output
            for (uint32_t i = 0; i < num_expecteds; ++i)
            {
              NNPR_ENSURE_STATUS(
                nnfw_train_set_expected(session, i, expected_data[i].data(), &expected_infos[i]));
            }

            // train
            measure.run(epoch, n, [&]() { NNPR_ENSURE_STATUS(nnfw_train(session, true)); });

            // store loss
            for (int32_t i = 0; i < num_expecteds; ++i)
            {
              float temp = 0.f;
              NNPR_ENSURE_STATUS(nnfw_train_get_loss(session, i, &temp));
              losses[i] += temp;
            }
          }

          // print loss
          std::cout << std::fixed;
          std::cout << "Epoch " << epoch + 1 << "/" << num_epoch;
          measure.printTimeMs(epoch, AggregateType::AVERAGE);
          std::cout.precision(4);
          std::cout << " - loss: ";
          for (uint32_t i = 0; i < num_expecteds; ++i)
          {
            std::cout << "[" << i << "] " << losses[i] / num_step;
          }
        }

        //
        // VALIDATION
        //
        // TODO Enable validation
        if (false)
        {
          std::fill(losses.begin(), losses.end(), 0);
          // TODO Use validation data length
          const int num_valid_step = data_length / tri.batch_size;
          for (uint32_t n = 0; n < num_valid_step; ++n)
          {
            // TODO get as much validation data as batchsize
            if (!generator(n, input_data, expected_data))
              break;

            // prepare input
            for (uint32_t i = 0; i < num_inputs; ++i)
            {
              NNPR_ENSURE_STATUS(
                nnfw_train_set_input(session, i, input_data[i].data(), &input_infos[i]));
            }

            // prepare output
            for (uint32_t i = 0; i < num_expecteds; ++i)
            {
              NNPR_ENSURE_STATUS(
                nnfw_train_set_expected(session, i, expected_data[i].data(), &expected_infos[i]));
            }

            // validation
            NNPR_ENSURE_STATUS(nnfw_train(session, false));

            // get validation loss and accuracy
            for (int32_t i = 0; i < num_expecteds; ++i)
            {
              float temp = 0.f;
              NNPR_ENSURE_STATUS(nnfw_train_get_loss(session, i, &temp));
              losses[i] += temp;
              // TODO get validation accuracy
            }
          }

          // print validation loss and accuracy
          std::cout << std::fixed;
          std::cout.precision(4);
          std::cout << " - val_loss: ";
          for (uint32_t i = 0; i < num_expecteds; ++i)
          {
            std::cout << "[" << i << "] " << losses[i] / num_valid_step;
          }
          // TODO print validation accuracy
        }

        std::cout << std::endl;
      }
    });

    if (args.getExportModelFilename() != "")
      NNPR_ENSURE_STATUS(nnfw_train_export_circle(session, args.getExportModelFilename().c_str()));

    NNPR_ENSURE_STATUS(nnfw_close_session(session));

    measure.printResult();

    return 0;
  }
  catch (boost::program_options::error &e)
  {
    std::cerr << "E: " << e.what() << std::endl;
    exit(-1);
  }
  catch (std::runtime_error &e)
  {
    std::cerr << "E: Fail to run by runtime error:" << e.what() << std::endl;
    exit(-1);
  }
}
