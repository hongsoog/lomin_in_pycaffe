# save detection v2 mode parameters into npy files
import inspect
import logging
import sys

import torch
import numpy as np

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list

# for model debugging log
from model_log import  logger

def pytorch_model_load(log_into_file=False):
    """load pytorch detection v2 models
    Args:
        none

    Returns:
        models, transforms
    """
    from maskrcnn_benchmark.config import cfg
    # log file patth: ./load_backbone_body_parms_log.txt
    if log_into_file:
        my_name = inspect.currentframe().f_code.co_name
        log_file_path = f"./log/{my_name}_log.txt"
        original_std_out = sys.stdout
        f = open(log_file_path, 'w')
        sys.stdout = f

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

    # model version
    version = "v2"

    # test image file path
    image_file_path = "../sample_images/detection/1594202471809.jpg"

    #config_file = os.path.join('../model/detection', detect_model[version]["config_file"])
    config_file = f"../model/detection/" + detect_model[version]["config_file"]
    #weight_file = os.path.join('../model/detection', detect_model[version]["weight_file"])
    param_file = f"../model/detection/" + detect_model[version]["weight_file"]
    print(f"config_file: {config_file}")
    print(f"param_file: {param_file}")

    is_recognition = False
    # clone project level config and merge with experiment config

    cfg = cfg.clone()
    cfg.merge_from_file(config_file)

    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model.to(device)

    # set to evaluation mode for interference
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir='/dev/null')
    _ = checkpointer.load(param_file)

    # build_transforms defined in maskrcnn_benchmark.data.transforms/*.py
    transforms = build_transforms(cfg, is_recognition)
    cpu_device = torch.device("cpu")
    score_thresh = cfg.TEST.SCORE_THRESHOLD

    if log_into_file:
        sys.stdout = original_std_out
        f.close()

    return transforms, model

# --------------------------------------------------------
# pytorch detection v2 model parameter saving functions
# --------------------------------------------------------
def save_pytorch_model_non_learnable_parameters(model, log_into_file=False):
    """Save pytorch detection model's FronzenBatchNorm2d parameters
    Args:
        model: returned mode from pytorch_model_load()
    Returns:
        none
    """
    # log file patth: ./load_backbone_body_parms_log.txt
    if log_into_file:
        my_name = inspect.currentframe().f_code.co_name
        log_file_path = f"./log/{my_name}_log.txt"
        original_std_out = sys.stdout
        f = open(log_file_path, 'w')
        sys.stdout = f

    # ----------------------------------
    # parameters of FrozenBatchNorm2d (fixed values)
    # ----------------------------------
    print("\n\n")
    print("-" * 80)
    print("saving non-learnalbe parameters of FrozenBatchNorm2d in detection v2 model")
    print("-" * 80)
    itr = model.named_buffers()  # get iterator

    for buffer_name, buffer in itr:
        file_name = f"./npy_save/{buffer_name.replace('.', '_')}"
        t_list = buffer_name.split('.')

        if t_list[-1] == 'weight' or t_list[-1] == 'bias':

            # convert torch in cpu to numpy ndarray
            if buffer.requires_grad:
                arr = buffer.detach().cpu().numpy()

            else:
                arr = buffer.cpu().numpy()

            np.save(file_name, arr)
            print(f"{buffer_name} of {arr.shape}\n\t\tsaved in {file_name}.npy")

    if log_into_file:
        sys.stdout = original_std_out
        f.close()


def save_pytorch_model_learnable_parameters(model, log_into_file=False):
    """Save pytorch detection model's learable layer parameters
       main Conv2d, Maxpool
    Args:
        model: returned mode from pytorch_model_load()
    Returns:
        none
    """
    # log file patth: ./load_backbone_body_parms_log.txt
    if log_into_file:
        my_name = inspect.currentframe().f_code.co_name
        log_file_path = f"./log/{my_name}_log.txt"
        original_std_out = sys.stdout
        f = open(log_file_path, 'w')
        sys.stdout = f

    # ----------------------------------
    # learnable parameters of Conv2d
    # ----------------------------------
    print("-" * 80)
    print("saving learnalbe parameters of layers in detection v2 model")
    print("-" * 80)
    itr = model.named_parameters()  # get iterator

    for param_name, param in itr:
        file_name = f"./npy_save/{param_name.replace('.', '_')}"

        if param.requires_grad:
            arr = param.detach().cpu().numpy()
        else:
            arr = param.cpu().numpy()

        np.save(file_name, arr)
        print(f"{param_name} of {arr.shape}\n\t\tsaved in {file_name}.npy")

    if log_into_file:
        sys.stdout = original_std_out
        f.close()


def save_pytorch_model_parameters(model, log_into_file=False):
    # save Conv2d parameters
    save_pytorch_model_learnable_parameters(model, log_into_file=log_into_file)

    # save FrozenBatchNora2d parameters
    save_pytorch_model_non_learnable_parameters(model, log_into_file=log_into_file)


if __name__ == "__main__":

    logger.setLevel(logging.CRITICAL)
    transform, detection_model = pytorch_model_load()

    save_pytorch_model_parameters(detection_model, log_into_file=True)


