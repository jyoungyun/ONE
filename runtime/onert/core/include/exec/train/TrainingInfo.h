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

#ifndef __ONERT_EXEC_TRAIN_TRAINING_INFO_H__
#define __ONERT_EXEC_TRAIN_TRAINING_INFO_H__

namespace onert
{
namespace exec
{
namespace train
{

class TrainingInfo
{
public:
  TrainingInfo() : _batch_size(1), _iteration(0) {}
  TrainingInfo(const TrainingInfo &obj) = default;
  TrainingInfo(TrainingInfo &&) = default;
  TrainingInfo &operator=(const TrainingInfo &) = default;
  TrainingInfo &operator=(TrainingInfo &&) = default;
  ~TrainingInfo() = default;

  uint32_t batchSize() const { return _batch_size; }
  void setBatchSize(const uint32_t batch_size) { _batch_size = batch_size; }
  uint32_t iteration() const { return _iteration; }
  void setIteration(const uint32_t iteration) { _iteration = iteration; }

private:
  uint32_t _batch_size;
  uint32_t _iteration;  
};

} // namespace train
} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_TRAIN_TRAINING_INFO_H__
