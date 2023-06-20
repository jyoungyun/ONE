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

#ifndef __ONERT_TRAIN_RAWDATALOADER_H__
#define __ONERT_TRAIN_RAWDATALOADER_H__

#include "allocation.h"
#include "nnfw.h"

#include <string>
#include <fstream>

namespace onert_train
{
class RawDataLoader
{
public:
  RawDataLoader(nnfw_session *sess) : _session(sess) {}
  Generator loadDatas(const std::string &input_file, const std::string &expected_file,
                      std::vector<nnfw_tensorinfo> &input_infos,
                      std::vector<nnfw_tensorinfo> &output_infos);

private:
  nnfw_session *_session;
  std::ifstream _input_file;
  std::ifstream _expected_file;
};
} // namespace onert_train

#endif // __ONERT_TRAIN_RAWDATALOADER_H__
