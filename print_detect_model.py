#!/home/kimkk/miniconda3/envs/lomin/bin/python

import os
import argparse

import torch
import torch.onnx

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer


class DetectionDemo(object):
    # --------------------------------
    # __init__(cfg, weight, is_recognition=False)
    # --------------------------------
    def __init__(self, model_cfg, weight, is_recognition=False):
        self.is_recognition = is_recognition
        self.cfg = model_cfg.clone()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model = build_detection_model(self.cfg)
        self.model.to(self.device)

        # set to evaluation mode for interference
        self.model.eval()

        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir='/dev/null')
        _ = checkpointer.load(weight)

        # build_transforms defined in  maskrcnn_benchmark.data.transforms/*.py
        self.transforms = build_transforms(self.cfg, self.is_recognition)
        self.cpu_device = torch.device("cpu")
        self.score_thresh = self.cfg.TEST.SCORE_THRESHOLD


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


if __name__ == '__main__':

    # command line arguments definition
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Lomin OCR Detection Model printer")

    parser.add_argument('--version', choices=['v1', 'v2'], default='v2',
                        help='set detection model version')

    # command line argument parsing
    args = parser.parse_args()
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

    # Detection model creation
    demo = DetectionDemo(cfg, weight_file)

    # print loaded model
    print(f"==== demo.model ==== {demo.model}")

    print(f"==== demo.model.backbone ==== {demo.model.backbone}")
    print(f"==== demo.model.rpn ==== {demo.model.rpn}")
