# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor
from maskrcnn_benchmark.structures.image_list import to_image_list

#for model debugging loe
import logging
from model_log import logger

class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n====================================== RPNHeadConvRegressor.__init__ BEGIN")
            logger.debug(f"\t// Params:")
            logger.debug(f"\t\tin_channels: {in_channels}  - number of ch. of the input feature")
            logger.debug(f"\t\tnum_anchors: {num_anchors}  - number of anchors to be predicted")
            logger.debug(f"\n\tsuper(RPNHeadConvRegressor, self).__init__()")
            logger.debug(f"\tself.cls_logits = nn.Conv2d(in_channles={in_channels}, num_anchors= {num_anchors}, kernel_size=1, stride=1)")
            logger.debug(f"self.bbox_pred = nn.Conv2d(")
            logger.debug(f"\t\tin_channels={in_channels}, {num_anchors} * 4, kernel_size = 1, stride = 1\n\n")

        super(RPNHeadConvRegressor, self).__init__()

        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d( in_channels, num_anchors * 4, kernel_size=1, stride=1 )

        if logger.level == logging.DEBUG:
            logger.debug(f"\tfor l in [self.cls_logits, self.bbox_pred]:")

        for l in [self.cls_logits, self.bbox_pred]:

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\ttorch.nn.init.normal_(l.weight, std=0.01)")
                logger.debug(f"\t\ttorch.nn.init.constant_(l.bias, 0)")

            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNHeadConvRegressor.__init__ END")

    def forward(self, x):
        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNHeadConvRegressor.forward(sefl, x) BEGIN")
            logger.debug(f"\t// Params:")
            logger.debug(f"\tx.type = {x.type}")
            logger.debug(f"\tlen(x) = {len(x)}")

        if logger.level == logging.DEBUG:
            print(f"assert isinstance(x, (list, tuple))")
            print(f"logits = [self.cls_logits(y) for y in x]")
            print(f"bbox_reg = [self.bbox_pred(y) for y in x]")

        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        if logger.level == logging.DEBUG:
            logger.debug(f"logits:\n")
            logger.debug(f"{logits}")
            logger.debug(f"box_reg:\n")
            logger.debug(f"{bbox_reg}")
            logger.debug(f"return logits, bbox_reg")
            logger.debug(f"\n====================================== RPNHeadConvRegressor.forward(sefl, x) END")

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNHeadFeatureSingleConv.__init__(self, cfg, in_channels) BEGIN")
            logger.debug(f"Param: in_channles: {in_channels}")
            logger.debug(f"super(RPNHeadFeatureSingleConv, self).__init__()")
            logger.debug(f"self.conv = nn.Conv2d( in_channels, in_channels, kernel_size=3, stride=1, padding=1)")

        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        if logger.level == logging.DEBUG:
            logger.debug(f"for l in [self.conv]:")

        for l in [self.conv]:
            if logger.level == logging.DEBUG:
                logger.debug(f"l: {l}")
                logger.debug(f"torch.nn.init.normal_(l.weight, std=0.01)")
                logger.debug(f"torch.nn.init.constant_(l.bias, 0)")
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

        if logger.level == logging.DEBUG:
            logger.debug("self.out_channels = in_channels")
            logger.debug(f"\n====================================== RPNHeadFeatureSingleConv.__init__(self, cfg, in_channels) END")

    def forward(self, x):
        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNHeadFeatureSingleConv.forward(self, x) BEGIN")
            logger.debug(f"Param:")
            logger.debug(f"\tx.type : {x.type}")
            logger.debug(f"\tlen(x) : {len(x)}")
            logger.debug(f"assert isinstance(x, (list, tuple))")
            logger.debug(f"x = [F.relu(self.conv(z)) for z in x]")

        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        if logger.level == logging.DEBUG:
            logger.debug(f"\tx.type : {x.type}")
            logger.debug(f"\tlen(x) : {len(x)}")
            logger.debug(f"return x")
            logger.debug(f"\n====================================== RPNHeadFeatureSingleConv.forward(self, x) END")
        return x


@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNHead.__init__(self, cfg, in_channels, num_anchaors) BEGIN")
            logger.debug(f"\t// Params:")
            logger.debug(f"\t\tin_channels: {in_channels}")
            logger.debug(f"\t\tnum_anchors: {num_anchors}")
            logger.debug(f"super(RPNHead, self).__init__()")

        super(RPNHead, self).__init__()

        if logger.level == logging.DEBUG:
            logger.debug(f"self.conv = nn.Conv2d( in_channels, in_channels, kernel_size=3, stride=1, padding=1)")
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        import pdb; pdb.set_trace()

        if logger.level == logging.DEBUG:
            logger.debug(f"self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)")
            logger.debug(f"self.bbox_pred = nn.Conv2d( in_channels, num_anchors * 4, kernel_size=1, stride=1)")

        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        if logger.level == logging.DEBUG:
           logger.debug(f"for l in [self.conv, self.cls_logits, self.bbox_pred]:")

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            if logger.level == logging.DEBUG:
                logger.debug(f"l: {l}")
                logger.debug(f"torch.nn.init.normal_(l.weight, std=0.01)")
                logger.debug(f"torch.nn.init.constant_(l.bias, 0)")

            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNHead.__init__(self, cfg, in_channels, num_anchors) END")

    def forward(self, x):
        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNHead.forward(self, x) BEGIN")
            logger.debug(f"logits = []")
            logger.debug(f"bbox_reg = []")

        logits = []
        bbox_reg = []

        if logger.level == logging.DEBUG:
            logger.debug(f"for feature in x:")

        for feature in x:
            if logger.level == logging.DEBUG:
                logger.debug(f"feature: {feature}")
                logger.debug(f"self.conv: {self.conv}")
                logger.debug(f"t = F.relu(self.conv(feature))")

            t = F.relu(self.conv(feature))

            if logger.level == logging.DEBUG:
                logger.debug(f"t: {t}")
                logger.debug(f"self.cls_logits: {self.cls_logits}")
                logger.debug(f"logits.append(self.cls_logits(t))")

            logits.append(self.cls_logits(t))

            if logger.level == logging.DEBUG:
                logger.debug(f"t: {t}")
                logger.debug(f"self.bbox_pred: {self.bbox_pred}")
                logger.debug(f"bbox_reg.append(self.bbox_pred(t))")

            bbox_reg.append(self.bbox_pred(t))

        if logger.level == logging.DEBUG:
            logger.debug(f"return logits, bbox_reg")
            logger.debug(f"\n====================================== RPNHead.forward(self, x) END")

        return logits, bbox_reg

class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs 
    RPN proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):

        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNModule.__init__(self, cfg, in_channels) BEGIN")
            logger.debug(f"Param:")
            logger.debug(f"\tin_channels: {in_channels}")
            logger.debug(f"super(RPNModule, self).__init__()")

        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        if logger.level == logging.DEBUG:
            logger.debug("anchor_generator = make_anchor_generator(cfg)")

        anchor_generator = make_anchor_generator(cfg)

        if logger.level == logging.DEBUG:
            logger.debug(f"rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]")

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]

        if logger.level == logging.DEBUG:
            logger.debug(f"head = rpn_head( cfg, in_channels, anchor_generator.num_anchors_per_location()[0] )")

        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        if logger.level == logging.DEBUG:
            logger.debug(f"rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))")
            logger.debug(f"box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder)")

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder)

        if logger.level == logging.DEBUG:
            logger.debug(f"self.anchor_generator = anchor_generator")
            logger.debug(f"self.head = head")
            logger.debug(f"self.box_selector_test = box_selector_test")

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test

        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNModule.__init__(self, cfg, in_channels) END")

    def forward(self, images, features, targets=None):

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
        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNModule.forward(self, images, features, targets=None) BEGIN")
            logger.debug(f"objectness, rpn_box_regression = self.head(features)")
            logger.debug(f"anchors = self.anchor_generator(images, features)")

        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if logger.level == logging.DEBUG:
            logger.debug(f"return self._forward_test(anchors, objectness, rpn_box_regression)")
        return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        if logger.level == logging.DEBUG:
            logger.debug(f"\n====================================== RPNModule._forward_test(self, anchors, objectness, rpn_box_regression) BEGIN")
            logger.debug(f"\t// Params:")
            logger.debug(f"\t\tanchors: {anchors}")
            logger.debug(f"\t\tobjectness: {objectness}")
            logger.debug(f"\t\trpn_box_regression: {rpn_box_regression}")

        if logger.level == logging.DEBUG:
            logger.debug(f"boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)")


            boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            if logger.level == logging.DEBUG:
                logger.debug(f"if self.cfg.MODEL.RPN_ONLY == {self.cfg.MODEL.RPN_ONLY}:")
                logger.debug("\tFor end-to-end models, the RPN proposals are an intermediate state")
                logger.debug("\tand don't bother to sort them in decreasing score order. For RPN-only")
                logger.debug("\tmodels, the proposals are the final output and we return them in")
                logger.debug("\thigh-to-low confidence order.\n")

                logger.debug(f'\tinds = [ box.get_field("objectness").sort(descending=True)[1] for box in boxes ]')
                logger.debug(f"\tboxes = [box[ind] for box, ind in zip(boxes, inds)]")
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]

        if logger.level == logging.DEBUG:
            logger.debug(f"return boxes, {{ }}")
            logger.debug(f"\n====================================== RPNModule._forward_test(self, anchors, objectness, rpn_box_regression) END")
        return boxes, {}


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)
    return RPNModule(cfg, in_channels)
