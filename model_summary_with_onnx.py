#!/home/kimkk/miniconda3/envs/lomin/bin/python

import os
import argparse
from pprint import pprint
from glob import glob
from tqdm import tqdm

from PIL import Image, ImageDraw

import torch
import torch.onnx

import numpy as np

#from torchinfo import summary
import torchinfo

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.label_catalog import LabelCatalog
from maskrcnn_benchmark.utils.converter import Converter


class DetectionDemo(object):
    #--------------------------------
    # __init__(cfg, weight, is_recognition=False)
    #--------------------------------
    def __init__(self, cfg, weight, is_recognition=False):
        self.is_recognition = is_recognition
        self.cfg = cfg.clone()
        self.device = torch.device(cfg.MODEL.DEVICE)
        #self.device = torch.device("cpu")
        self.model =  build_detection_model(self.cfg)
        self.model.to(self.device)

        # set to evaluation mode for interference
        self.model.eval()

        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir='/dev/null')
        _ = checkpointer.load(weight)

        # build_transforms defined in  maskrcnn_benchmark.data.transforms/*.py
        self.transforms = build_transforms(self.cfg, self.is_recognition)
        self.cpu_device = torch.device("cpu")
        self.score_thresh = self.cfg.TEST.SCORE_THRESHOLD

    #--------------------------------
    # run_on_pil(image_origin)
    #--------------------------------
    def run_on_pil_image(self, image_origin):
        # pil_image defined in __main__
        prediction = self.compute_prediction(pil_image)
        prediction = self.filter_by_score(prediction)
        prediction = prediction.resize(image_origin.size)
        result = self.parse_result(prediction)
        return result


    #--------------------------------
    # compute_predicion(image)
    #--------------------------------
    def compute_prediction(self, image):

        print(f"PIL image.size: {image.size}")
        # convert PIL Image to torch Tensor
        image_tensor = self.transforms(image)
        print(f"image_tesnor.shape: {image_tensor.shape}")

        image_list = to_image_list(image_tensor, self.cfg.DATALOADER.SIZE_DIVISIBILITY).to(self.device)
        print(f"image_list.tensors.shape: {image_list.tensors.shape}")
        print(f"len(image_list.tensors): {len(image_list.tensors)}")
        print(f"image_list.image_sizes.shape: {image_list.image_sizes}")
        print(f"len(image_list.image_sizes): {len(image_list.image_sizes)}")

        # dummy_input creation - still got error
        # https://github.com/pytorch/pytorch/issues/33443  xsidneib
        dummy_input = torch.randn(1, 3, 480, 576).to(self.device)

        # if mask rcnn does not detect any mask, so for some layer the input size 0,
        # use another iamge from the training set solve the problems
        with torch.no_grad():
            #pred = self.model(image_list)
            #pred = self.model(image_list.tensors)
            #torch.onnx.export(self.model, dummy_input, f"detection_model_{version}.onnx")
            torchinfo.summary(self.model,
                          input_data = [image_list.tensors],
                          col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
                          verbose=2)

            #torch.onnx.export(self.model, image_list.tensors, f"detection_model_{version}.onnx")
            pred = pred[0].to(self.cpu_device)

        return pred


    #--------------------------------
    # filter_by_score(predition)
    #--------------------------------
    def filter_by_score(self, prediction):
        filter_thres = prediction.get_field('scores') > self.score_thresh
        return prediction[filter_thres]

    #--------------------------------
    # parse_result(pred)
    #--------------------------------
    def parse_result(self, pred):
        bbox = pred.bbox.numpy().tolist()
        scores = pred.get_field('scores').numpy().tolist()
        #labels = pred.get_field('labels').numpy().tolist()
        return dict(
            bboxes=bbox,     # list of [x1, y1, x2, y2]
            #labels=labels,  # no interest on label
            scores=scores,
        )


# detection model conf and weidht file names
detect_model = {
        "v1" :
        {
            "config_file" : "config_det_v1_200723_001_180k.yaml",
            "weight_file" : "model_det_v1_200723_001_180k.pth"

        },
        "v2" :
        {
            "config_file" : "config_det_v2_200924_002_180k.yaml",
            "weight_file" : "model_det_v2_200924_002_180k.pth"
        }
}



if __name__ == '__main__':

    # command line arguments definition
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     description = "Lomin OCR Detection Model ONNX exporter")

    parser.add_argument('--version', choices=['v1', 'v2'], default = 'v2',
                        help='set detection model version')
    parser.add_argument('--image',
                        required=True,
                        help='input image file path')

    # command line argument parsing
    args =  parser.parse_args()
    version = args.version
    image_file_path = args.image

    # set model conf file path and mode weight file path
    # prefixed by ./model/[detection|recognition]
    config_file = os.path.join('model/detection', detect_model[version]["config_file"])
    weight_file = os.path.join('model/detection', detect_model[version]["weight_file"])

    # print cfg file path and weight file path (DEBUG)
    print(f"config file path: {config_file}")
    print(f"weight file path: {weight_file}")

    # clone project level config and merge with experiment config
    cfg = cfg.clone()
    cfg.merge_from_file(config_file)

    # Detection model creation
    demo = DetectionDemo(cfg, weight_file)

    # open image file as PIL.Image
    pil_image = Image.open(image_file_path).convert('RGB')
    org_pil_image = np.array(pil_image)

    prediction = demo.run_on_pil_image(org_pil_image)

