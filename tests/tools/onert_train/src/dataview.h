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

#ifndef __ONERT_TRAIN_DATAVIEW_H__
#define __ONERT_TRAIN_DATAVIEW_H__

#include "allocation.h"

#include <memory>
#include <vector>

namespace onert_train
{

typedef std::vector<Allocation>::iterator vec_iter;

class DataView
{
public:
  DataView(std::vector<Allocation> &data) : _data(std::move(data)), _it(_data.begin()) {}

  void setBatchSize(uint32_t batch_size) { _batch_size = batch_size; }

  // TODO Do not copy vector data
  std::vector<Allocation> fetch()
  {
    const auto _prev = _it;
    _it += _batch_size;
    return std::move(std::vector<Allocation>(_prev, _it));
  }

  void reset() { _it = _data.begin(); }

private:
  uint32_t _batch_size;
  std::vector<Allocation> _data;
  vec_iter _it;
};
} // namespace onert_train

#endif // __ONERT_TRAIN_DATAVIEW_H__
