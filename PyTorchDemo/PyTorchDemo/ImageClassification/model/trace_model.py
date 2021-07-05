import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

#model = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
#model.eval()
#example = torch.rand(1, 3, 224, 224)
#traced_script_module = torch.jit.trace(model, example)
#torchscript_model_optimized = optimize_for_mobile(traced_script_module)
#torchscript_model_optimized.save("mobilenet_quantized.pt")

model = torch.hub.load('yangsenius/TransPose:main', 'tpr_a4_256x192', pretrained=True)
model.eval()
example = torch.rand(1,3,256,192)
torchscript_model = torch.jit.trace(model, example)
#torchscript_model_optimized_transPose = optimize_for_mobile(torchscript_model)
#torchscript_model_optimized_transPose.save("mobilenet_transPose2.pt")
torch.jit.save(torchscript_model,"unoptimized_transPose.pt")
print("done")

#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#model.eval()
#torchscript_model = torch.jit.script(model)

#torchscript_model_optimized_transPose = optimize_for_mobile(torchscript_model)
#torchscript_model_optimized_transPose.save("mobilenet_boxmodel.pt")
#print("done")
