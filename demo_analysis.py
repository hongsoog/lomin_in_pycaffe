#!/home/kimkk/miniconda3/envs/lomin/bin/python

import os
import argparse
from pprint import pprint
from datetime import datetime
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.label_catalog import LabelCatalog
from maskrcnn_benchmark.utils.converter import Converter 


class DetectionDemo(object):
    def __init__(self, cfg, weight, is_recognition=False):
        self.is_recognition = is_recognition
        self.cfg = cfg.clone()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model =  build_detection_model(self.cfg)
        self.model.to(self.device)
        self.model.eval()

        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir='/dev/null')
        _ = checkpointer.load(weight)

        self.transforms = build_transforms(self.cfg, self.is_recognition)
        self.cpu_device = torch.device("cpu")
        self.score_thresh = self.cfg.TEST.SCORE_THRESHOLD

    def run_on_pil_image(self, image_origin):
        prediction = self.compute_prediction(image)
        prediction = self.filter_by_score(prediction)
        prediction = prediction.resize(image_origin.size)
        result = self.parse_result(prediction)
        return result

    def compute_prediction(self, image):
        image = self.transforms(image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY).to(self.device)
        with torch.no_grad():
            pred = self.model(image_list)
            pred = pred[0].to(self.cpu_device)
        return pred
    
    def filter_by_score(self, prediction):
        filter_thres = prediction.get_field('scores') > self.score_thresh
        return prediction[filter_thres]

    def parse_result(self, pred):
        bbox = pred.bbox.numpy().tolist()
        scores = pred.get_field('scores').numpy().tolist()
        labels = pred.get_field('labels').numpy().tolist()
        return dict(
            bboxes=bbox,
            labels=labels,
            scores=scores,
        )
    

class RecognitionDemo(DetectionDemo):
    def __init__(self, cfg, weight):
        self.batch_max_length = cfg.MODEL.TEXT_RECOGNIZER.BATCH_MAX_LENGTH
        self.load_converter(cfg)
        super(RecognitionDemo, self).__init__(cfg ,weight, True)

    def run_on_pil_image(self, image_origin):
        prediction = self.compute_prediction(image)
        encoded_text = prediction.get_field('pred')
        decoded_text = self.decode_text(encoded_text)
        return decoded_text

    def load_converter(self, cfg):
        characters = LabelCatalog.get(cfg.MODEL.TEXT_RECOGNIZER.CHARACTER)
        self.converter = Converter(characters)
    
    def decode_text(self, encoded_text):
        text = self.converter.decode(encoded_text)
        text = text[:text.find('[s]')]
        return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, choices=['detection', 'recognition'])
    args = parser.parse_args()
    ocr_type = args.type

    cfg_file = os.path.join('model', ocr_type, 'config.yaml')
    weight_file = os.path.join('model', ocr_type, 'model.pth')
    cfg = cfg.clone()
    cfg.merge_from_file(cfg_file)

    demo = {
        'detection': DetectionDemo,
        'recognition': RecognitionDemo
    }[ocr_type](cfg, weight_file)

    input_dir = os.path.join('sample_images', ocr_type)
    imglist = sorted([_ for _ in os.listdir(input_dir)
        if os.path.splitext(_)[1].lower() in ['.jpg', '.png', '.jpeg', '.tif']])

    for imgname in tqdm(imglist):
        print(imgname)
        image = Image.open(os.path.join(input_dir, imgname)).convert('RGB')
        prediction = demo.run_on_pil_image(image)
        pprint(prediction)
