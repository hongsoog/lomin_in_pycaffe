# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import PIL
from PIL import Image
import torch
import torchvision

from torchvision.transforms import functional as F

# added by kimkk for model visualization in tensorboard
#from torch.utils.tensorboard import SummaryWriter
import torchvision

# default `log_dir` is "runs"
# writer = SummaryWriter('runs/lomin_detect')

from maskrcnn_benchmark.structures.bounding_box import BoxList

# for model debugging log
from model_log import  logger

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        logger.debug(f"\n\t\ttransforms.py Compose class __call__  ====== BEGIN")
        logger.debug(f"\n\t\tfor t in self.transforms:")
        for t in self.transforms:
            logger.debug(f"\t\t\timage = {t}(image)")
            image = t(image)
        logger.debug(f"\n\t\treturn image")
        logger.debug (f"\t\ttransforms.py Compose class __call__  ====== END")

        return image

class Resize(object):
    def __init__(self, mode, min_sizes, max_size, fixed_size, target_interpolation):
        # mode
        # - detection model: keep_ratio
        # - recognition model: horizontal_padding
        self.mode = mode

        # min_sizes, max_size
        # - detection model: 480, 640
        # - recognition model: 800, 1333
        self.min_sizes = min_sizes
        self.max_size = max_size

        # fixed_size
        # - detection model: (-1, -1)
        # - recognition model:  (128, 32)
        self.fixed_size = fixed_size

        # target_interpolation
        # - detection model: bilinear
        # - recognition model: bilinear
        self.target_interpolation = target_interpolation

    def get_size(self, image_size):
        w, h = image_size  # h, w : PIL Image input image h, w
                           # h: 438, w: 512
        oh, ow = -1, -1    # oh, ow : output image width, height

        # i) recognition model: 'horizontal_padding'
        if self.mode == 'horizontal_padding':
            ow, oh = self.fixed_size
            target_width = int(w *(oh/h))
            if target_width < oh:
               target_width = oh
            if target_width > ow:
                target_width = ow
            ow = target_width

        # ii) detection model: 'keep_ratio'
        elif self.mode == 'keep_ratio':
            min_size = random.choice(self.min_sizes)  # self.min_sizes = 480
            max_size = self.max_size                  # self.max_size = 600
            min_original_size = float(min((w, h)))    # min_original_size = 438
            max_original_size = float(max((w, h)))    # max_original_size = 512

            # summary
            # take smaller one from height or width, and resize smaller one to 480
            # and larger one is resized while keeping ratio

            # i) first determine max_size
            #   max_size : min_size  = max_original_size : min_original_size  -- (1)
            #       ? :  480  =  512 : 438
            # from (1) max_size = max_original_size * min_size / min_orignal_size
            #                   =  480*512/438 = 531.09 = 561
            # max size= 561.095
            max_size = max_original_size / min_original_size * min_size

            # max size= min(561.095, 600) = 561.095
            max_size = min(max_size, self.max_size)

            # ii) determine min_size from the determined max_size
            #   max_size : min_size  = max_original_size : min_original_size  -- (2)
            #      561.095  :  ?  =  512 : 438
            # from (2) min_size  =  max_size * min_original_size /  max_original_size
            #                    = 561.095 * 438 /512 = 479.99 = round(479.99) = 480
            min_size = round(max_size * min_original_size / max_original_size)

            # vertical image
            #   ow = min_size, oh = max_size
            # horizontal image
            #   ow = max_size, oh = min_size
            ow = min_size if w < h else max_size

            oh = max_size if w < h else min_size

            # oh : 480, ow = 561.095
            # int() cause round off
            # oh : 480, ow = 561,
            # (438, 512)  => (480, 561)   ; keep ratio = 1.168
        return (int(oh), int(ow))

    def __call__(self, image):

        # code for tensorboard
        # grid = torchvision.utils.make_grid(image)
        # writer.add_image("input image to self.backbone", image.to_tensor(), 0)

        size = self.get_size(image.size)
        # resize(img: PIL Image (or Tensor), Size:List[int],
        #         interpolation: InterpolationMode = InterpolationMode.BILINEAR, max_size=None)
        # size : (h, w) format
        # return: resized PIL Image
        image = F.resize(image, size)
        return image

class ToTensor(object):
    # convert PIL.Image input to tensor object
    def __call__(self, image):
        return F.to_tensor(image)

class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True, to_n1p1=False):
        self.mean = mean      # cfg.INPUT.PIXEL_MEAN: [102.9801, 115.9465, 122.7717]
        self.std = std        # cfg.INPUT.PIXEL_STD: [1.0, 1.0, 1.0]
        self.to_bgr255 = to_bgr255
        self.to_n1p1 = to_n1p1

    def __call__(self, image):
        if image.shape[0] == 1:
            # 1-ch image tensor, convert it 3-ch image tensor by repeating
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            # 4-ch image tensor, convert it 3-ch image tensor by taking first 3 channels
            image = image[:3]
        if self.to_bgr255:
            # default: convert BGR255 format
            torch.set_printoptions(profile="full")
            #logger.debug(f"before to_bgr255:\n{image}")
            image = image[[2, 1, 0]] * 255
        elif self.to_n1p1:
            # n1p1: what's this?
            image.sub_(0.5).div_(0.5)

        # mean and std in normalize
        #logger.debug(f"mean: {self.mean}, std: {self.std} after from RGB to BGR")
        #logger.debug(f"before normalization:\n{image}")
        image = F.normalize(image, mean=self.mean, std=self.std)
        #logger.debug(f"after normalization:\n{image}")
        return image
