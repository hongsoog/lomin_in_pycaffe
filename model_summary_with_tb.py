#!/home/kimkk/miniconda3/envs/lomin/bin/python

import os
import argparse
import torch
import torchvision
from PIL import Image

# added by kimkk for model visualization in tensorboard
#from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs"
#writer = SummaryWriter('runs/lomin_detect')

import numpy as np

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list


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

        # build_transforms defined in maskrcnn_benchmark.data.transforms/*.py
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
        """
        :param
        image: PIL.Image

        :return:
        pred
        """
        """
        :param image: 
        :return: 
        """

        image_tensor = self.transforms(image)
        grid = torchvision.utils.make_grid(image_tensor)
        writer.add_image("image after transforms", grid, 0)

        image_list = to_image_list(image_tensor, self.cfg.DATALOADER.SIZE_DIVISIBILITY).to(self.device)
        print(f"image_list.tensors.shape: {image_list.tensors.shape}")
        print(f"image_list.image_sizes: {image_list.image_sizes}")

        """
        torch.save(self.model, "./detection_model_v2.pth")
        torch.save(self.model.backbone, "./detection_model_v2_backbone.pth")
        torch.save(self.model.backbone.body, "./detection_model_v2_backbone_body.pth")
        torch.save(self.model.backbone.fpn, "./detection_model_v2_backbone_fpn.pth")
        """

        with torch.no_grad():
            pred = self.model(image_list)
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
                                     description = "Lomin OCR Detection Model GraphViz exporter")

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

    prediction = demo.run_on_pil_image(pil_image)

    # tb log
    writer.add_text("pil_image_height", "hello")
    writer.add_text("pil_image_height", str(pil_image.height))
    writer.add_text("pil_image_width", str(pil_image.width))
    writer.add_text("pil_image_mode", str(pil_image.mode))

    print(prediction)
