/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_SUPPORT_EXAMPLE_TASK_TEXT_DESKTOP_NL_CLASSIFIER_OP_RESOLVER_H_
#define TENSORFLOW_LITE_SUPPORT_EXAMPLE_TASK_TEXT_DESKTOP_NL_CLASSIFIER_OP_RESOLVER_H_

#include <memory>

#include "tensorflow/lite/op_resolver.h"

namespace tflite {
namespace task {
namespace text {

std::unique_ptr<tflite::OpResolver> CreateCustomOpResolver();

}  // namespace text
}  // namespace task
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_EXAMPLE_TASK_TEXT_DESKTOP_NL_CLASSIFIER_OP_RESOLVER_H_
