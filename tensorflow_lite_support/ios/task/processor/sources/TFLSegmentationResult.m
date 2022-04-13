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
#import "tensorflow_lite_support/ios/task/processor/sources/TFLSegmentationResult.h"

@implementation TFLCategoryMask

- (instancetype)initWithWidth:(NSInteger)width height:(NSInteger)height mask:(UInt8 *)mask {
  self = [super init];
  if (self) {
    _width = width;
    _height = height;
    if (mask != NULL) {
      _mask = malloc(width * height * sizeof(UInt8));
      memcpy(_mask, mask, width * height * sizeof(UInt8));
    }
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  return [[TFLCategoryMask alloc] initWithWidth:self.width
                                         height:self.height
                                           mask:self.mask];
}

- (void)dealloc {
  free(self.mask);
}

@end

@implementation TFLConfidenceMask

- (instancetype)initWithWidth:(NSInteger)width height:(NSInteger)height mask:(float *)mask {
  self = [super init];
  if (self) {
    _width = width;
    _height = height;
    if (mask != NULL) {
      _mask = malloc(width * height * sizeof(float));
      memcpy(_mask, mask, width * height * sizeof(float));
    }
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  return [[TFLConfidenceMask alloc] initWithWidth:self.width
                                           height:self.height
                                             mask:self.mask];
}

- (void)dealloc {
  free(self.mask);
}

@end

@implementation TFLColoredLabel

- (instancetype)initWithRed:(NSUInteger)r green:(NSUInteger)g blue:(NSUInteger)b label:(NSString *)label displayName:(NSString *)displayName {
  self = [super init];
  if (self) {
    _r = r;
    _g = g;
    _b = b;
    _label = [label copy];
    _displayName = [displayName copy];
  }
  return self;
}

@end

@implementation TFLSegmentation

- (instancetype)initWithConfidenceMasks:(NSArray<TFLConfidenceMask *> *)confidenceMasks  coloredLabels:(NSArray<TFLColoredLabel *> *)coloredLabels {
  return [self initWithConfidenceMasks:confidenceMasks categoryMask:nil coloredLabels:coloredLabels];
}

- (instancetype)initWithCategoryMask:(TFLCategoryMask *)categoryMask coloredLabels:(NSArray<TFLColoredLabel *> *)coloredLabels {
  return [self initWithConfidenceMasks:nil categoryMask:categoryMask coloredLabels:coloredLabels];
}

- (instancetype)initWithConfidenceMasks:(NSArray<TFLConfidenceMask *> *)confidenceMasks categoryMask:(TFLCategoryMask *)categoryMask coloredLabels:(NSArray<TFLColoredLabel *> *)coloredLabels {
  self = [super init];
  if (self) {
    _confidenceMasks = [confidenceMasks copy];
    _categoryMask = [categoryMask cop];
    _coloredLabels = [coloredLabels copy];
  }
  return self;
}

@end

@implementation TFLSegmentationResult
- (instancetype)initWithSegmentations:(NSArray<TFLSegmentation *> *)segmentations {
  self = [super init];
  if (self) {
    _segmentations = [segmentations copy];
  }

  return self;
}
@end
