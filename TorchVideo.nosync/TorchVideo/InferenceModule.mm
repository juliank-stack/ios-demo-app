// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#import "InferenceModule.h"
#import <LibTorch/LibTorch.h>

const int INPUT_WIDTH = 192;
const int INPUT_HEIGHT = 256;
const int OUTPUT_SIZE = 52223;


@implementation InferenceModule {
    @protected torch::jit::script::Module _impl;
}

NSInteger arrayWithIndexSort(NSArray* first, NSArray* second, void* context)
{
    id firstValue = [first objectAtIndex:0];
    id secondValue = [second objectAtIndex:0];
    return [secondValue compare:firstValue];
}


- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::load(filePath.UTF8String);
            _impl.eval();
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (int)argMax:(NSArray*)array {
    int maxIdx = 0;
    float maxVal = -FLT_MAX;
    for (int j = 0; j < OUTPUT_SIZE; j++) {
      if ([array[j] floatValue]> maxVal) {
          maxVal = [array[j] floatValue];
          maxIdx = j;
      }
    }
    return maxIdx;
}


- (NSArray<NSNumber*>*)classifyFrames:(void*)framesBuffer {
    try {
        at::Tensor tensor = torch::from_blob(framesBuffer, {1, 3, 256, 192}, at::kFloat);
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        
        auto outputTensor = _impl.forward({ tensor }).toTensor();

        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        
        NSMutableArray* scores = [[NSMutableArray alloc] init];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
          [scores addObject:@(floatBuffer[i])];
        }
        float xmax = -MAXFLOAT;
        float xmin = MAXFLOAT;
        for (NSNumber *num in scores) {
            float x = num.floatValue;
            if (x < xmin) xmin = x;
            if (x > xmax) xmax = x;
        }
        
//        NSMutableArray* scoresIdx = [[NSMutableArray alloc] init];
//        for (int i = 0; i < OUTPUT_SIZE; i++) {
//            [scoresIdx addObject:[NSArray arrayWithObjects:scores[i], @(i), nil]];
//        }
//
//        NSArray* sortedScoresIdx = [scoresIdx sortedArrayUsingFunction:arrayWithIndexSort context:NULL];
//
//        NSMutableArray* results = [[NSMutableArray alloc] init];
//        for (int i = 0; i < TOP_COUNT; i++)
//            [results addObject: sortedScoresIdx[i][1]];
        
        return [scores copy];
        
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end
