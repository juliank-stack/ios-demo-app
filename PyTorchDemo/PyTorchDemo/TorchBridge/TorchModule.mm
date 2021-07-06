#import "TorchModule.h"
#import <LibTorch/LibTorch.h>
#import <opencv2/core/core_c.h>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/imgproc/imgproc_c.h>
#import <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


@implementation TorchModule {
 @protected
  torch::jit::script::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
  self = [super init];
  if (self) {
    try {
      auto qengines = at::globalContext().supportedQEngines();
      if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
        at::globalContext().setQEngine(at::QEngine::QNNPACK);
      }
      _impl = torch::jit::load(filePath.UTF8String);
      _impl.eval();
    } catch (const std::exception& exception) {
      NSLog(@"%s", exception.what());
      return nil;
    }
  }
  return self;
}

@end

@implementation VisionTorchModule
/*- (cv::Mat) matFromImageBuffer: (CVImageBufferRef) buffer {

    cv::Mat mat ;

    CVPixelBufferLockBaseAddress(buffer, 0);

    void *address = CVPixelBufferGetBaseAddress(buffer);
    int width = (int) CVPixelBufferGetWidth(buffer);
    int height = (int) CVPixelBufferGetHeight(buffer);

    mat   = cv::Mat(height, width, CV_8UC4, address, 0);
    //cv::cvtColor(mat, _mat, CV_BGRA2BGR);

    CVPixelBufferUnlockBaseAddress(buffer, 0);

    return mat;
}*/
- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
  try {
      
      //cv::Mat img;
      //img = cv::Mat(256, 192, CV_32F, imageBuffer);

      //cv::cvtColor(mat, _mat, CV_BGRA2BGR);
      //std::cout << img.at<float>(0,2) << std::endl;
      //img.convertTo( img, CV_32FC3, 1/255.0 );
      at::Tensor tensor = torch::from_blob(imageBuffer, {3,256,192}, at::kFloat);
      
      
      
      
      
      
    at::Tensor unsqueezed = tensor.unsqueeze(0);
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    
    // Erwartete Tensorgröße {1,17,64,48}
    at::Tensor outputTensor = _impl.forward({unsqueezed}).toTensor();
      //int width = 48;
      //int height = 64;
      //cv::Mat outputMat(cv::Size{height,width}, CV_32F, outputTensor.data_ptr<float>());
      //UIImage* test = MatToUIImage(outputMat);
    //std::cout << tensor << std::endl;
    //std::cout << outputTensor << std::endl;
    float* floatBuffer = outputTensor.data_ptr<float>();
    if (!floatBuffer) {
      return nil;
    }
    NSMutableArray* results = [[NSMutableArray alloc] init];
    for (int i = 0; i < 52223; i++) {
        //std::cout << floatBuffer[i] << std::endl;
      [results addObject:@(floatBuffer[i])];
    }
    return [results copy];
  } catch (const std::exception& exception) {
    NSLog(@"%s", exception.what());
  }
  return nil;
}



@end

@implementation NLPTorchModule

- (NSArray<NSNumber*>*)predictText:(NSString*)text {
  try {
    const char* buffer = text.UTF8String;
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::Tensor tensor = torch::from_blob((void*)buffer, {1, (int64_t)(strlen(buffer))}, at::kByte);
    auto outputTensor = _impl.forward({tensor}).toTensor();
    float* floatBuffer = outputTensor.data_ptr<float>();
    if (!floatBuffer) {
      return nil;
    }
    NSMutableArray* results = [[NSMutableArray alloc] init];
    for (int i = 0; i < 16; i++) {
      [results addObject:@(floatBuffer[i])];
    }
    return [results copy];
  } catch (const std::exception& exception) {
    NSLog(@"%s", exception.what());
  }
  return nil;
}

- (NSArray<NSString*>*)topics {
  try {
    auto genericList = _impl.run_method("get_classes").toList();
    NSMutableArray<NSString*>* topics = [NSMutableArray<NSString*> new];
    for (int i = 0; i < genericList.size(); i++) {
      std::string topic = genericList.get(i).toString()->string();
      [topics addObject:[NSString stringWithCString:topic.c_str() encoding:NSUTF8StringEncoding]];
    }
    return [topics copy];
  } catch (const std::exception& exception) {
    NSLog(@"%s", exception.what());
  }
  return nil;
}

@end
