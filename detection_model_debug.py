# pytorch 및 mask_rcnn 관련 import
import os
import argparse
from pprint import pprint
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

import torch
from torchviz import make_dot
from torchvision import models
from pprint import pprint

import numpy as np

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.label_catalog import LabelCatalog
from maskrcnn_benchmark.utils.converter import Converter

# caffe glog 로깅 레벨 설정 및 caffe 및 tools 관련 import
os.environ['GLOG_minloglevel'] = '3'
import caffe
import tools.solvers
import tools.lmdb_io
import tools.prototxt
import tools.pre_processing

# for model debugging log
from model_log import  logger

# Detection V2 Model in PyTorch
class DetectionDemo(object):
    # --------------------------------
    # __init__(cfg, weight, is_recognition=False)
    # --------------------------------
    def __init__(self, cfg, weight, is_recognition=False):
        self.is_recognition = is_recognition
        self.cfg = cfg.clone()
        self.device = torch.device(cfg.MODEL.DEVICE)
        # self.device = torch.device("cpu")
        self.model = build_detection_model(self.cfg)
        self.model.to(self.device)

        # set to evaluation mode for interference
        self.model.eval()

        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir='/dev/null')
        _ = checkpointer.load(weight)

        # build_transforms defined in maskrcnn_benchmark.data.transforms/*.py
        self.transforms = build_transforms(self.cfg, self.is_recognition)
        self.cpu_device = torch.device("cpu")
        self.score_thresh = self.cfg.TEST.SCORE_THRESHOLD

    # --------------------------------
    # run_on_pil(image_origin)
    # --------------------------------
    def run_on_pil_image(self, image_origin):
        # pil_image defined in __main__
        # call detection/recognition mode with PIL image
        prediction = self.compute_prediction(pil_image)

        prediction = self.filter_by_score(prediction)

        # prediction result (bbox) adjust to fit original image size
        prediction = prediction.resize(image_origin.size)
        result = self.parse_result(prediction)
        return result

    # --------------------------------
    # compute_predicion(image)
    # --------------------------------
    def compute_prediction(self, image):

        # input image transformation
        # - resize: using get_size() calc the image resize so as to use input to model
        # - to_tensor: image to tensonr( numpy.ndarray)
        # - normalization: RGB to GBR color channel ordering
        #       multiply 255 on pixel value
        #       normalization with constant mean and constant std defined in cfg
        logger.debug(f"compute_prediction(self, image)")
        logger.debug(f"\n\timage: H, W=({image.height},{image.width})")
        logger.debug(f"\n\timage_tensor = self.transforms(image)")

        image_tensor = self.transforms(image)

        import numpy as np
        np.save("./npy_save/transformed_tensor.npy", image_tensor)

        logger.debug(f"\n\timage_tensor.shape: {image_tensor.shape}")


        # padding image for 32 divisible size on width and height
        logger.debug(f"\n\tpadding images for 32 divisible size on width and height")
        logger.debug(f"\timage_list = to_image_list(image_tensor, {self.cfg.DATALOADER.SIZE_DIVISIBILITY}).to(self.device)")

        image_list = to_image_list(image_tensor, self.cfg.DATALOADER.SIZE_DIVISIBILITY).to(self.device)

        np.save("./npy_save/padded_tensor.npy", image_list.tensors.cpu())

        logger.debug(f"\timage_list.image_sizes: {image_list.image_sizes}")
        logger.debug(f"\timage_list.tensors.shape: {image_list.tensors.shape}")

        torch.save(self.model, "./detection_model_v2.pth")
        torch.save(self.model.backbone, "./detection_model_v2_backbone.pth")
        torch.save(self.model.backbone.body, "./detection_model_v2_backbone_body.pth")
        torch.save(self.model.backbone.fpn, "./detection_model_v2_backbone_fpn.pth")

        with torch.no_grad():
            logger.debug(f"\tpred = self.model(image_list)")
            pred = self.model(image_list)
            pred = pred[0].to(self.cpu_device)

        """
        model_param_dict = dict(self.model.named_parameters())
        logger.debug("-"*80)

        for key, value in model_param_dict.items():
            logger.debug(key)
            logger.debug(value)
            logger.debug("-"*80)
        """

        # make_dot(self.model(image_list), params=model_param_dict).render(f"detection_model_{version}", format="png")
        # https: // github.com / szagoruyko / pytorchviz / blob / master / examples.ipynb
        """
        with torch.onnx.set_training(self.model.backbone, False):
            trace, _ = torch.jit._get_trace_graph(self.model, args(image_list))
        make_dot_from_trace(trace).render(f"detection_mode_{version}_structure", format="png")
        """
        return pred

    # --------------------------------
    # filter_by_score(predition)
    # --------------------------------
    def filter_by_score(self, prediction):
        filter_thres = prediction.get_field('scores') > self.score_thresh
        return prediction[filter_thres]

    # --------------------------------
    # parse_result(pred)
    # --------------------------------
    def parse_result(self, pred):
        bbox = pred.bbox.numpy().tolist()
        scores = pred.get_field('scores').numpy().tolist()
        # labels = pred.get_field('labels').numpy().tolist()
        return dict(
            bboxes=bbox,  # list of [x1, y1, x2, y2]
            # labels=labels,  # no interest on label
            scores=scores,
        )


# detection model conf and weight file names
detect_model = {
    "v1":
        {
            "config_file": "config_det_v1_200723_001_180k.yaml",
            "weight_file": "model_det_v1_200723_001_180k.pth"

        },
    "v2":
        {
            "config_file": "config_det_v2_200924_002_180k.yaml",
            "weight_file": "model_det_v2_200924_002_180k.pth"
        }
}


def bb_image_draw(pil_image, line_color=(0, 0, 255), line_width=4, score_threshold=0.5):
    pil_image_cp = pil_image.copy()

    # for drawing bbox
    draw = ImageDraw.Draw(pil_image_cp)

    num_bbox_included = 0

    for idx, bbox in enumerate(bboxes):
        if scores[idx] > score_threshold:
            num_bbox_included += 1

            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

            draw.rectangle(((x1, y1), (x2, y2)), outline=line_color, width=line_width)

    return pil_image_cp, num_bbox_included

# model version
version = "v2"

# test image file path
image_file_path = "./sample_images/detection/1594202471809.jpg"
#image_file_path = "./sample_images/detection/1596537103856.jpeg"
#image_file_path = "./sample_images/video_frames/frame000000.png"

# set model conf file path and mode weight file path
# prefixed by ./model/[detection|recognition]
config_file = os.path.join('./model/detection', detect_model[version]["config_file"])
weight_file = os.path.join('./model/detection', detect_model[version]["weight_file"])

# clone project level config and merge with experiment config
cfg = cfg.clone()
cfg.merge_from_file(config_file)

# Detection model object creation
demo = DetectionDemo(cfg, weight_file)

# open image file as PIL.Image with RGB
pil_image = Image.open(image_file_path).convert('RGB')
org_pil_image = np.array(pil_image)
prediction = demo.run_on_pil_image(pil_image)


# draw with predicted boxes
bboxes = prediction['bboxes']
scores = prediction['scores']


bboxed_image, num_boxes = bb_image_draw(pil_image, line_color=(255, 0, 0), line_width = 3, score_threshold = 0.3)

# Display an image with Python
# https://stackoverflow.com/questions/35286540/display-an-image-with-python
plt.figure()
plt.imshow(bboxed_image)
plt.show()

# Detection model Info
