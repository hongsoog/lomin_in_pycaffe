import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import  make_retinanet_postprocessor
from ..anchor_generator import make_anchor_generator_retinanet

from maskrcnn_benchmark.modeling.box_coder import BoxCoder


# for model debugging log
from maskrcnn_benchmark.structures.image_list import ImageList
import logging
from model_log import  logger

class RetinaNetHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n=========================================== RetinaNetHead._init__(cfg, in_channels): BEGIN")

        super(RetinaNetHead, self).__init__()
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                        * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels,  num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )

        # Initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits,
                  self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)


        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n=========================================== RetinaNetHead._init__(cfg, in_channels): END")

    def forward(self, x):
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n=========================================== RetinaNetHead.forward(self, x): BEGIN")
            logger.debug(f"\tParam:")
            logger.debug(f"\t\tlen(x)): {len(x)} x is features returned from FPN")  # x is features returned from FPN
            for idx, f in enumerate(x):
                logger.debug(f"\t\t\tx[{idx}].shape: {f.shape}")
            logger.debug(f"logits = []")
            logger.debug(f"bbox_reg = []\n")

        logits = []
        bbox_reg = []

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\nfor feature in x:")

        for idx, feature in enumerate(x):

            if logger.level == logging.DEBUG:
                logger.debug(f"\t===== iteration: {idx} ====")
                logger.debug(f"\tfeature[{idx}].shape: {feature.shape}\n")
                logger.debug(f"\tcls_tower => cls_logits => logits[]")
                logger.debug(f"\tlogits.append(self.cls_logits(self.cls_tower(feature)))\n")
                logger.debug(f"\tbbox_tower => bbox_pre => bbox_reg[]")
                logger.debug(f"\tbbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))\n")

            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))

        if logger.level == logging.DEBUG:
            logger.debug(f" ==== logits ====")
            for idx, l in enumerate(logits):
                logger.debug(f"logits[{idx}].shape: {l.shape}")

            logger.debug(f"\n ==== bbox_reg ====")
            for idx, b in enumerate(bbox_reg):
                logger.debug(f"bbox_reg[{idx}].shape: {b.shape}")

            logger.debug(f"\nreturn logits, bbox_reg")
            logger.debug(f"\n\n=========================================== RetinaNetHead.forward(self, x): END")


        return logits, bbox_reg


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n=========================================== RetinaNetModule.__init__(self, cfg, in_channels): BEGIN")
            logger.debug(f"\tsuper(RetinaNetModule, self).__init__()")
            logger.debug(f"\tself.cfg = cfg.clone()")

        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        if logger.level == logging.DEBUG:
            logger.debug(f"\tanchor_generator = make_anchor_generator_retinanet(cfg)\n")

        anchor_generator = make_anchor_generator_retinanet(cfg)

        if logger.level == logging.DEBUG:
            logger.debug(f"head = RetinaNetHead(cfg, in_channels={in_channels})")

        head = RetinaNetHead(cfg, in_channels)

        if logger.level == logging.DEBUG:
            logger.debug("box_coder = BoxCoder(weights=(10., 10., 5., 5.))")

        box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        if logger.level == logging.DEBUG:
            logger.debug("box_selector_test = make_retinanet_postprocessor(cfg, box_coder)")

        box_selector_test = make_retinanet_postprocessor(cfg, box_coder)

        if logger.level == logging.DEBUG:
            logger.debug("self.anchor_generator = anchor_generator")
            logger.debug("self.head = head")
            logger.debug("self.box_selector_test = box_selector_test")

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n=========================================== RetinaNetModule.__init__(self, cfg, in_channels): END")

    def forward(self, images, features, targets=None):
    #def forward(self, rpn_inputs):

        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        # added for tensforboard debugging BEGIN
        #images_sizes, images_tensors, features, targets = rpn_inputs
        #images = ImageList(images_tensors, images_sizes)
        # added for tensforboard debugging END

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n=========================================== RetinaNetModule.forward(self, images, features, targets=None): BEGIN")
            logger.debug(f"\tParams:")
            logger.debug(f"\t\ttype(images.image_size): {type(images.image_sizes)}")
            logger.debug(f"\t\ttype(images.tensors): {type(images.tensors)}")
            logger.debug(f"\t\tlen(features)): {len(features)}")
            for idx, f in enumerate(features):
               logger.debug(f"\t\t\tfeature[{idx}].shape: {f.shape}")

        #------------------------
        #  1. head
        #------------------------
        if logger.level == logging.DEBUG:
            logger.debug(f"self.head: {self.head}")
            logger.debug(f"box_cls, box_regression = self.head(features)")

        box_cls, box_regression = self.head(features)

        #------------------------
        #  2. anchor_generator
        #------------------------
        if logger.level == logging.DEBUG:
            logger.debug(f"self.anchor_generator: {self.anchor_generator}")
            logger.debug(f"anchors = self.anchor_generator(images, features)")

        anchors = self.anchor_generator(images, features)

        #------------------------
        #  3. _forward_test
        #------------------------
        if self.training:
            if logger.level == logging.DEBUG:
                logger.debug(f"if self.training == {self.training}")
                logger.debug(f"\treturn self._forward_train(anchors, box_cls, box_regression, targets)")
                logger.debug(f"\n\n=========================================== RetinaNetModule.forward(self, images, features, targets=None): END")
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            if logger.level == logging.DEBUG:
                logger.debug(f"if self.training == {self.training}")
                logger.debug(f"\treturn self._forward_test(anchors, box_cls, box_regression)")
                logger.debug(f"\n\n=========================================== RetinaNetModule.forward(self, images, features, targets=None): END")
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_test(self, anchors, box_cls, box_regression):

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n=========================================== RetinaNetModule._forward_test(self, anchors, box_cls, box_regression): BEGIN")
            logger.debug(f"params:")    # anchors is list
            logger.debug(f"\tlen(anchors)\n: {len(anchors)}")    # anchors is list
            logger.debug(f"\tlen(box_cls)\n: {len(box_cls)}")
            logger.debug(f"\tlen(box_regression): {len(box_regression)}")
            logger.debug(f"self.box_selector_test: {self.box_selector_test}")
            logger.debug(f"boxes = self.box_selector_test(anchors, box_cls, box_regression)")

        #------------------------
        #  4. box_selector_test
        #------------------------
        boxes = self.box_selector_test(anchors, box_cls, box_regression)

        if logger.level == logging.DEBUG:
            logger.debug(f"len(boxes): {len(boxes)}")
            logger.debug(f"return boxes, {{}} # {{}} is just empty dictionayr")
            logger.debug(f"\n\n=========================================== RetinaNetModule._forward_test(self, anchors, box_cls, box_regression): END")

        # {} is just empty dictionary if self.training == FALSE
        return boxes, {}


def build_retinanet(cfg, in_channels):
    return RetinaNetModule(cfg, in_channels)
