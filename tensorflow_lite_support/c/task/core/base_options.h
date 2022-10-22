/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_SUPPORT_C_TASK_CORE_BASE_OPTIONS_H_
#define TENSORFLOW_LITE_SUPPORT_C_TASK_CORE_BASE_OPTIONS_H_

#include <stdbool.h>
#include <stdint.h>

// Defines C Structs for Base Options Shared by all tasks.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Holds cpu settings.
typedef struct TfLiteCpuSettings {
  // Specifies the number of threads to be used for TFLite
  // ops that support multi-threading when running inference with CPU.
  // num_threads should be greater than 0 or equal to -1. Setting num_threads to
  // -1 has the effect to let TFLite runtime set the value.
  int num_threads;
} TfLiteCpuSettings;

// The device type for Core ML Delegate.
typedef enum CoreMLDelegateSettingsEnabledDevices {
  // Always create Core ML delegate.
  kDevicesAll = 0,
  // Create Core ML delegate only on devices with Apple Neural Engine.
  kDevicesWithNeuralEngine = 1,
} CoreMLDelegateSettingsEnabledDevices;

// Holds Core ML Delegate settings.
typedef struct TfLiteCoreMLDelegateSettings {
  // Enables Core ML Delegate.
  bool enable_delegate;

  /** The device set to enable Core ML Delegate. */
  CoreMLDelegateSettingsEnabledDevices enabled_devices;

  /** Specifies target Core ML version for model conversion.
   * If not set to one of the valid versions (2, 3), the delegate will use the
   * highest version possible in the platform.
   */
  int32_t coreml_version;
} TfLiteCoreMLDelegateSettings;

typedef enum TfLiteCoralSettingsPerformance {
  kPerformanceUndefined = 0,
  kPerformanceMaximum = 1,
  kPerformanceHigh = 2,
  kPerformanceMedium = 3,
  kPerformanceLow = 4,
} TfLiteCoralSettingsPerformance;

// Holds Coral Edge TPU Delegate settings.
typedef struct TfLiteCoralSettings {
  // Enables the Coral delegate.
  bool enable_delegate;
  // The Edge Tpu device to be used. See
  // https://github.com/google-coral/libcoral/blob/982426546dfa10128376d0c24fd8a8b161daac97/coral/tflite_utils.h#L131-L137
  const char* device;
  // The desired performance level. This setting adjusts the internal clock
  // rate to achieve different performance / power balance. Higher performance
  // values improve speed, but increase power usage.
  TfLiteCoralSettingsPerformance performance;
  // If true, always perform device firmware update (DFU) after reset. DFU is
  // usually only necessary after power cycle.
  bool usb_always_dfu;
  // The maximum bulk in queue length. Larger queue length may improve USB
  // performance on the direction from device to host. When not specified (or
  // zero), `usb_max_bulk_in_queue_length` will default to 32 according to the
  // current EdgeTpu Coral implementation.
  int32_t usb_max_bulk_in_queue_length;
} TfLiteCoralSettings;

// Holds settings for one possible acceleration configuration.
typedef struct TfLiteComputeSettings {
  // Holds cpu settings.
  TfLiteCpuSettings cpu_settings;
  // Holds Core ML Delegate settings.
  TfLiteCoreMLDelegateSettings coreml_delegate_settings;
  // Holds Coral Edge TPU Delegate settings.
  TfLiteCoralSettings coral_delegate_settings;
} TfLiteComputeSettings;

// Represents external files used by the Task APIs (e.g. TF Lite Model File).
// For now you can only specify the path of the file using file_path:
// In future other sources may be supported.
typedef struct TfLiteExternalFile {
  // The path to the file to open.
  const char* file_path;
  // Additional option for byte data when it's supported.
} TfLiteExternalFile;

// Holds the base options that is used for creation of any type of task. It has
// fields withh important information acceleration configuration, tflite model
// source etc.
// This struct must be zero initialized before setting any options as this
// will result in seg faults.
typedef struct TfLiteBaseOptions {
  // The external model file, as a single standalone TFLite file. It could be
  // packed with TFLite Model Metadata[1] and associated files if exist. Fail to
  // provide the necessary metadata and associated files might result in errors.
  // Check the documentation for each task about the specific requirement.
  // [1]: https://www.tensorflow.org/lite/convert/metadata
  TfLiteExternalFile model_file;

  // Holds settings for one possible acceleration configuration
  // including.cpu/gpu settings. Please see documentation of
  // TfLiteComputeSettings and its members for more details.
  TfLiteComputeSettings compute_settings;
} TfLiteBaseOptions;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_C_TASK_CORE_BASE_OPTIONS_H_
