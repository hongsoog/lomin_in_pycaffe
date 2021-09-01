import torch
import torchvision
import onnx

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
x = [torch.rand(3, 800, 800)]
torch.onnx.export(model, x, "mask_rcnn.onnx", opset_version = 11)

onnx_model = onnx.load("mask_rcnn.onnx")
onnx.checker.check_model(onnx_model)
onnx.shape_inference.infer_shapes(onnx_model)
