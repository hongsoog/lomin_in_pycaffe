import math
import inspect
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
            logger.debug(f"\n\n\t\t\tRetinaNetHead.__init__(cfg, in_channels) {{ //BEGIN")
            logger.debug(f"\t\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")
            logger.debug(f"\t\t\t\tParams:")
            logger.debug(f"\t\t\t\t\tcfg:")
            logger.debug(f"\t\t\t\t\tin_channles: {in_channels}:")

        super(RetinaNetHead, self).__init__()

        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        logger.debug(f"\t\t\t\tnum_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1")
        logger.debug(f"\t\t\t\t>>num_classes: {num_classes}")

        logger.debug(f"\t\t\t\t>> cfg.MODEL.RETINANET.ASPECT_RATIOS: {cfg.MODEL.RETINANET.ASPECT_RATIOS}")
        logger.debug(f"\t\t\t\t>> cfg.MODEL.RETINANET.SCALES_PER_OCTAVE: {cfg.MODEL.RETINANET.SCALES_PER_OCTAVE}")
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                        * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
        logger.debug(f"\t\t\t\tnum_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \\")
        logger.debug(f"\t\t\t\t                * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE")
        logger.debug(f"\t\t\t\t>>num_anchors: {num_anchors}")

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append( nn.Conv2d( in_channels, in_channels, kernel_size=3, stride=1, padding=1 ) )
            cls_tower.append(nn.ReLU())

            bbox_tower.append( nn.Conv2d( in_channels, in_channels, kernel_size=3, stride=1, padding=1 ) )
            bbox_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d( in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1 )
        self.bbox_pred = nn.Conv2d( in_channels,  num_anchors * 4, kernel_size=3, stride=1, padding=1 )

        # Initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)


        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n}} // END RetinaNetHead._init__(cfg, in_channels)")

    def forward(self, x):
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n\tRetinaNetHead.forward(self, x) {{ // BEGIN")
            logger.debug(f"\t// // defined in {inspect.getfile(inspect.currentframe())}")
            logger.debug(f"\t\tParam:")
            logger.debug(f"\t\t\tlen(x)): {len(x)}")  # x is features returned from FPN
            for idx, f in enumerate(x):
                logger.debug(f"\t\t\t\tx[{idx}].shape: {f.shape}")

        logits = []
        bbox_reg = []

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\tlogits = []")
            logger.debug(f"\t\tbbox_reg = []\n")

            logger.debug(f"\t\tself.cls_tower:\n{self.cls_tower}\n")
            logger.debug(f"\t\tself.cls_logits:\n{self.cls_logits}\n")
            logger.debug(f"\t\tself.bbox_tower:\n{self.bbox_tower}\n")
            logger.debug(f"\t\tself.bbox_pred:\n{self.bbox_pred}\n")

            logger.debug(f"\n\n\t\tfor idx, feature in enumerate(x) {{")

        for idx, feature in enumerate(x):

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t\t===== iteration: {idx} ====")
                logger.debug(f"\t\t\tfeature[{idx}].shape: {feature.shape}\n")

            logits.append(self.cls_logits(self.cls_tower(feature)))
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t\tlogits.append(self.cls_logits(self.cls_tower(feature)))")
                logger.debug(f"\t\t\t\tlen(logits): {len(logits)}\n")

            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t\tbbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))")
                logger.debug(f"\t\t\t\tlen(bbox_reg): {len(bbox_reg)}\n")


        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n\t\t}}// END for idx, feature n enumerate(x)")

        if logger.level == logging.DEBUG:
            logger.debug(f" ==== logits ====")
            for idx, l in enumerate(logits):
                logger.debug(f"logits[{idx}].shape: {l.shape}")

            logger.debug(f"\n ==== bbox_reg ====")
            for idx, b in enumerate(bbox_reg):
                logger.debug(f"bbox_reg[{idx}].shape: {b.shape}")

            logger.debug(f"\nreturn logits, bbox_reg")
            logger.debug(f"\t}} // END RetinaNetHead.forward(self, x)")

        return logits, bbox_reg


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\nRetinaNetModule.__init__(self, cfg, in_channels) {{ // BEGIN")
            logger.debug(f"// defined in {inspect.getfile(inspect.currentframe())}")
            logger.debug(f"\tParams:")
            logger.debug(f"\t\tcfg:")
            logger.debug(f"\t\tin_channels: {in_channels}\n")

            logger.debug(f"\tsuper(RetinaNetModule, self).__init__()")
            logger.debug(f"\tself.cfg = cfg.clone()")

        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        if logger.level == logging.DEBUG:
            logger.debug(f"\tanchor_generator = make_anchor_generator_retinanet(cfg)\n")

        anchor_generator = make_anchor_generator_retinanet(cfg)

        if logger.level == logging.DEBUG:
            logger.debug(f"\tanchor_generator: {anchor_generator}")
            logger.debug(f"\thead = RetinaNetHead(cfg, in_channels={in_channels})")

        head = RetinaNetHead(cfg, in_channels)

        if logger.level == logging.DEBUG:
            logger.debug(f"\thead: {head}")
            logger.debug(f"\tbox_coder = BoxCoder(weights=(10., 10., 5., 5.))")

        box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        if logger.level == logging.DEBUG:
            logger.debug(f"\tbox_coder: {box_coder}")
            logger.debug(f"\tbox_selector_test = make_retinanet_postprocessor(cfg, box_coder)")

        box_selector_test = make_retinanet_postprocessor(cfg, box_coder)

        if logger.level == logging.DEBUG:
            logger.debug(f"\tbox_selector_test: {box_selector_test}")
            logger.debug(f"\tself.anchor_generator = anchor_generator")
            logger.debug(f"\tself.head = head")
            logger.debug(f"\tself.box_selector_test = box_selector_test")

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n}} // RetinaNetModule.__init__(self, cfg, in_channels) END")
            logger.debug(f"\t}} // END build_retinanet(cfg, in_channels)")

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
            logger.debug(f"\n\nRetinaNetModule.forward(self, images, features, targets=None) {{ // BEGIN")
            logger.debug(f"// defined in {inspect.getfile(inspect.currentframe())}")
            logger.debug(f"\tParams:")
            logger.debug(f"\t\ttype(images.image_size): {type(images.image_sizes)}")
            logger.debug(f"\t\ttype(images.tensors): {type(images.tensors)}")
            logger.debug(f"\t\tlen(features)): {len(features)}")
            for idx, f in enumerate(features):
               logger.debug(f"\t\t\tfeature[{idx}].shape: {f.shape}")

        #------------------------
        #  1. head
        #------------------------
        box_cls, box_regression = self.head(features)
        if logger.level == logging.DEBUG:
            logger.debug(f"self.head: {self.head}")
            logger.debug(f"box_cls, box_regression = self.head(features)")
            logger.debug(f"\tlen(box_cls): {len(box_cls)}")
            for i, bc in enumerate(box_cls):
                logger.debug(f"\tbox_cls[{i}].shape: {bc.shape}")

            logger.debug(f"\tlen(box_regression): {len(box_regression)}")
            for i, br in enumerate(box_regression):
                logger.debug(f"\tbox_regression[{i}].shape: {br.shape}")

            """
            TODO : how to save list of tensor as numpy file
            https://www.aiworkbox.com/lessons/turn-a-list-of-pytorch-tensors-into-one-tensor 
            => not suitable of list of tensors, which have different shapes
            file_path = f"./npy_save/box_cls"
            arr = torch.stack(box_cls).numpy()
            np.save(file_path, arr)
            logger.debug(f"box_cls of shape {arr.shape} saved into {file_path}.npy\n\n")

            file_path = f"./npy_save/box_regression"
            arr = torch.stack(box_regression).numpy()
            np.save(file_path, arr)
            logger.debug(f"box_regression of shape {arr.shape} saved into {file_path}.npy\n\n")
            """

    #------------------------
    #  2. anchor_generator
    #------------------------
        anchors = self.anchor_generator(images, features)
        if logger.level == logging.DEBUG:
            logger.debug(f"anchors = self.anchor_generator(images, features)")
            logger.debug(f"self.anchor_generator: {self.anchor_generator}")
            logger.debug(f"anchors: {anchors}")


        #------------------------
        #  3. _forward_test
        #------------------------
        if self.training:
            if logger.level == logging.DEBUG:
                logger.debug(f"if self.training: {self.training}")
                logger.debug(f"\treturn self._forward_train(anchors, box_cls, box_regression, targets)")
                logger.debug(f"\t}} // END RetinaNetModule.forward(self, images, features, targets=None)")
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            if logger.level == logging.DEBUG:
                logger.debug(f"if self.training: {self.training}")
                logger.debug(f"\treturn self._forward_test(anchors, box_cls, box_regression)")
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_test(self, anchors, box_cls, box_regression):

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\nRetinaNetModule._forward_test(self, anchors, box_cls, box_regression) {{ // BEGIN")
            logger.debug(f"// defined in {inspect.getfile(inspect.currentframe())}")
            logger.debug(f"\tparams:")    # anchors is list
            logger.debug(f"\tlen(anchors): {len(anchors)}")    # anchors is list
            logger.debug(f"\tlen(box_cls): {len(box_cls)}")
            logger.debug(f"\tlen(box_regression): {len(box_regression)}")

            logger.debug(f"\tself.box_selector_test: {self.box_selector_test}")
            logger.debug(f"\tboxes = self.box_selector_test(anchors, box_cls, box_regression)")

        #------------------------
        #  4. box_selector_test
        #------------------------
        boxes = self.box_selector_test(anchors, box_cls, box_regression)

        if logger.level == logging.DEBUG:
            logger.debug(f"len(boxes): {len(boxes)}")
            logger.debug(f"(boxes): {boxes}")
            logger.debug(f"return boxes, {{}} # {{}} is just empty dictionayr")
            logger.debug(f"\n\n}} // RetinaNetModule._forward_test(self, anchors, box_cls, box_regression): END")
            logger.debug(f"}} // END RetinaNetModule.forward(self, images, features, targets=None)")

        # {} is just empty dictionary if self.training == FALSE
        return boxes, {}


def build_retinanet(cfg, in_channels):
    if logger.level == logging.DEBUG:
        logger.debug(f"\tbuild_retinanet(cfg, in_channels) {{ // BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\tParam:")
        logger.debug(f"\t\t\tcfg:")
        logger.debug(f"\t\t\tin_channels: {in_channels}")
        logger.debug(f"\treturn RetinaNetModule(cfg, in_channels)")

    return RetinaNetModule(cfg, in_channels)
