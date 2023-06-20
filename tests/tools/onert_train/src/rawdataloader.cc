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

#include "rawdataloader.h"
#include "nnfw_util.h"

#include <iostream>
#include <stdexcept>
#include <numeric>

namespace onert_train
{
Generator RawDataLoader::loadDatas(const std::string &input_file, const std::string &expected_file,
                                   std::vector<nnfw_tensorinfo> &input_infos,
                                   std::vector<nnfw_tensorinfo> &expected_infos)
{
  auto size_accumulator = [](const unsigned int &a, const nnfw_tensorinfo &b) {
    return a + bufsize_for(&b);
  };

  auto input_sample_size =
    std::accumulate(input_infos.begin(), input_infos.end(), 0, size_accumulator);
  auto expectd_sample_size =
    std::accumulate(expected_infos.begin(), expected_infos.end(), 0, size_accumulator);

  try
  {
    _input_file = std::ifstream(input_file, std::ios::ate | std::ios::binary);
    _expected_file = std::ifstream(expected_file, std::ios::ate | std::ios::binary);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    std::exit(-1);
  }

  return [input_sample_size, expectd_sample_size, input_infos, expected_infos,
          this](uint32_t idx, std::vector<Allocation> &inputs, std::vector<Allocation> &expecteds) {
    _input_file.seekg(idx * input_sample_size, std::ios::beg);

    for (uint32_t i = 0; i < input_infos.size(); ++i)
    {
      // allocate memory for data
      auto bufsz = bufsize_for(&input_infos[i]);
      _input_file.read(reinterpret_cast<char *>(inputs[i].data()), bufsz);
    }

    _expected_file.seekg(idx * expectd_sample_size, std::ios::beg);

    for (uint32_t i = 0; i < expected_infos.size(); ++i)
    {
      // allocate memory for data
      auto bufsz = bufsize_for(&expected_infos[i]);
      _expected_file.read(reinterpret_cast<char *>(expecteds[i].data()), bufsz);
    }

    return true;
  };
}

} // end of namespace onert_train
