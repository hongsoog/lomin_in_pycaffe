#!/home/kimkk/miniconda3/envs/lomin/bin/python

import os
import argparse
from pprint import pprint
from datetime import datetime
from glob import glob
from tqdm import tqdm

from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

import torch
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

        print (f"in run_on_pil_image, image size: {image.size}")
        prediction = self.compute_prediction(image)  # <== ?? where is image
        prediction = self.filter_by_score(prediction)
        prediction = prediction.resize(image_origin.size)
        result = self.parse_result(prediction)
        return result

    #--------------------------------
    # compute_predicion(image)
    #--------------------------------
    def compute_prediction(self, image):

        # image is type of PIL.Image
        print (f"in compute_prediction before transfor, image type: {type(image)}")
        print (f"in compute_prediction before transforms, image.size: {image.size}")
        #image.show()
        image = self.transforms(image)  # convert PIL Image to torch Tensor

        # image is type of torch.Tensor
        print (f"in compute_prediction after transforms, type(image): {type(image)}")
        print (f"in compute_prediction after transforms, image.shape: {image.shape}")
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY).to(self.device)


        print (f"in compute_prediction type(image_list): {type(image_list)}")
        print (f"in compute_prediction type(image_list.tensors): {type(image_list.tensors)}")
        print (f"in compute_prediction image_list.tensors.shape: {image_list.tensors.shape}")
        #print (f"in compute_prediction help(image_list): {help(image_list)}")

        """
        torchinfo.summary(model=self.model,
                          input_size=(1, 3, 480, 576),
                          input_data = image_list,
                          col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
                          verbose=2)
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
        labels = pred.get_field('labels').numpy().tolist()
        return dict(
            bboxes=bbox,
            labels=labels,
            scores=scores,
        )

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True, choices=['v1', 'v2'])
    args =  parser.parse_args()
    version = args.version

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

    # DetectionDemo
    demo = DetectionDemo(cfg, weight_file)


    # print model class name (DEBUG)
    #print(f"model class: {demo}")

    # print model cfg (DEBUG)
    #print(f"model cfg: {cfg}")

    #torchinfo.summary(demo.model, input_size=(1, 3, 640, 480), verbose=2)
    #torchinfo.summary(model=demo.model, verbose=1)
    #torchinfo.summary(model=demo.model, verbose=2)
    #print(demo.model)

    input_dir = os.path.join('sample_images', 'detection')
    imglist = sorted([_ for _ in os.listdir(input_dir)
        if os.path.splitext(_)[1].lower() in ['.jpg', '.png', '.jpeg', '.tif']])

    for imgname in tqdm(imglist):
        image = Image.open(os.path.join(input_dir, imgname)).convert('RGB')

        image.show()
        # debug input image info
        print (f"in main, image file path: {imgname}")
        print (f"in main, image size: {image.size}")

        prediction = demo.run_on_pil_image(image)

        draw = ImageDraw.Draw(image)

        bboxes = prediction['bboxes']
        scores = prediction['scores']
        print(f"socres: {scores}")

        num_bbox_included = 0
        for idx, bbox in enumerate(bboxes):

            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

            print(f"index: {idx}, scores[idx]: {scores[idx]}")
            if scores[idx] > 0.8:
                num_bbox_included += 1
                draw.rectangle( ((x1, y1), (x2, y2)), outline =(0, 0, 255), width = 4)

        np_image = np.array(image)
        plt.imshow(np_image)

        print(f"included_bbox/total_bbox_detected: {num_bbox_included}/{len(bboxes)}")
        plt.show()
        break
