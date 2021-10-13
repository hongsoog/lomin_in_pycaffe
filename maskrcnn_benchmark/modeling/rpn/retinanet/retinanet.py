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
            logger.debug(f"\t\t\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
            logger.debug(f"\t\t\t\t// Params:")
            logger.debug(f"\t\t\t\t\t// cfg:")
            logger.debug(f"\t\t\t\t\t//in_channles: {in_channels}\n")

        super(RetinaNetHead, self).__init__()

        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        logger.debug(f"\t\t\t\tnum_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1")
        logger.debug(f"\t\t\t\t// num_classes: {num_classes}")

        logger.debug(f"\t\t\t\t// cfg.MODEL.RETINANET.ASPECT_RATIOS: {cfg.MODEL.RETINANET.ASPECT_RATIOS}")
        logger.debug(f"\t\t\t\t// cfg.MODEL.RETINANET.SCALES_PER_OCTAVE: {cfg.MODEL.RETINANET.SCALES_PER_OCTAVE}")
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                        * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
        logger.debug(f"\t\t\t\tnum_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \\")
        logger.debug(f"\t\t\t\t                * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE")
        logger.debug(f"\t\t\t\t// num_anchors: {num_anchors}")

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
        logger.debug(f"\n\n\tRetinaNetHead.forward(self, x) {{ // BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t// Param:")
        logger.debug(f"\t\t\t// self: {self}")

        # x is features returned from FPN
        logger.debug(f"\t\t\t// len(x)): {len(x)} # x is features retruned fron FPN forward")
        logger.debug(f"\n\t\t\t\t# features info")
        for i, f in enumerate(x):
            if i < 3:
                logger.debug(f"\t\t\t\t// feature[{i}].shape: {f.shape} <== P{i + 2} after FPN")
            else:
                logger.debug(f"\t\t\t\t// feature[{i}].shape: {f.shape} <== P{i + 3} after FPN")

        logger.debug(f"\n")

        logits = []
        logger.debug(f"\t\tlogits = []")

        bbox_reg = []
        logger.debug(f"\t\tbbox_reg = []\n")

        logger.debug(f"\t\t#=========================================")
        logger.debug(f"\t\t# for every P (total 5) from FPN,")
        logger.debug(f"\t\t# - apply cls_tower and cls_logits")
        logger.debug(f"\t\t# - apply bbox_tower and bbox_pred")
        logger.debug(f"\t\t# hence 5 set of identical cls_tower, cls_logits, bbox_tower and bbox_pred")
        logger.debug(f"\t\t# should be prepared because caffe don't have sub-net iteration structure")
        logger.debug(f"\t\t#=========================================")

        logger.debug(f"\t\tfor idx, feature in enumerate(x) {{")

        total_num_iter = len(x)
        for idx, feature in enumerate(x):

            logger.debug(f"\t\t\t{{")
            logger.debug(f"\t\t\t# BEGIN iteration: {idx+1}/{total_num_iter}\n")
            logger.debug(f"\t\t\t// ===================================")
            if idx < 3:
                logger.debug(f"\t\t\t// feature P{idx+2} of shape: {feature.shape}")
            else:
                logger.debug(f"\t\t\t// feature p{idx+3} of shape: {feature.shape}")
            logger.debug(f"\t\t\t// ===================================")

            logger.debug(f"\t\t\t# 2-3-2-1-1. append cls_logits(cls_tower(feature))")
            logits.append(self.cls_logits(self.cls_tower(feature)))
            logger.debug(f"\t\t\tlogits.append(self.cls_logits(self.cls_tower(feature)))")
            logger.debug(f"\t\t\tlogits.append(self.cls_logits(self.cls_tower(feature)))")
            logger.debug(f"\t\t\t\t// feature.shape: {feature.shape}")
            logger.debug(f"\t\t\t\t// self.cls_tower: {self.cls_tower}")
            logger.debug(f"\t\t\t\t// self.cls_logits: {self.cls_logits}")
            logger.debug(f"\t\t\t\t// logits[-1].shape: {logits[-1].shape}")
            logger.debug(f"\t\t\t\t// len(logits): {len(logits)}\n")

            logger.debug(f"\t\t\t# 2-3-2-1-2. append bbox_pred(bbox_tower(feature))")
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
            logger.debug(f"\t\t\tbbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))")
            logger.debug(f"\t\t\t\t// feature.shape: {feature.shape}")
            logger.debug(f"\t\t\t\t// self.bbox_tower: {self.bbox_tower}")
            logger.debug(f"\t\t\t\t// self.bbox_pred: {self.bbox_pred}")
            logger.debug(f"\t\t\t\t// bbox_reg[-1].shape: {bbox_reg[-1].shape}")
            logger.debug(f"\t\t\t\t// len(bbox_reg): {len(bbox_reg)}\n")

            logger.debug(f"\t\t\t}} // END iteration: {idx+1}/{total_num_iter}\n")

        logger.debug(f"\n\n\t\t}}// END for idx, feature n enumerate(x)\n")

        logger.debug(f"\t\t// ==== logits ====")
        for idx, l in enumerate(logits):
            logger.debug(f"\t\t// logits[{idx}].shape: {l.shape}")

        logger.debug(f"\n\t\t// ==== bbox_reg ====")
        for idx, b in enumerate(bbox_reg):
            logger.debug(f"\t\t// bbox_reg[{idx}].shape: {b.shape}")

        logger.debug(f"\nreturn logits, bbox_reg")
        logger.debug(f"\t}} // END RetinaNetHead.forward(self, x)")

        return logits, bbox_reg


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):

        logger.debug(f"\t\tRetinaNetModule.__init__(self, cfg, in_channels) {{ // BEGIN")
        logger.debug(f"\t\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t\t// Params:")
        logger.debug(f"\t\t\t\t// cfg:")
        logger.debug(f"\t\t\t\t// in_channels: {in_channels}\n")

        logger.debug(f"\tsuper(RetinaNetModule, self).__init__()")
        super(RetinaNetModule, self).__init__()

        logger.debug(f"\tself.cfg = cfg.clone()")
        self.cfg = cfg.clone()

        logger.debug(f"\t#============================")
        logger.debug(f"\t# 1.2.1 anchor generator build")
        logger.debug(f"\t#============================")

        logger.debug(f"\tanchor_generator = make_anchor_generator_retinanet(cfg) // CALL\n\t{{")
        anchor_generator = make_anchor_generator_retinanet(cfg)
        logger.debug(f"\t}}\n\tanchor_generator = make_anchor_generator_retinanet(cfg) // RETURNED\n")
        logger.debug(f"\t// anchor_generator: {anchor_generator}")

        logger.debug(f"\t#============================")
        logger.debug(f"\t# 1.2.2 RPN head build ")
        logger.debug(f"\t#============================")
        logger.debug(f"\thead = RetinaNetHead(cfg, in_channels={in_channels}) // CALL\n\t{{")
        head = RetinaNetHead(cfg, in_channels)
        logger.debug(f"\t}}\n\thead = RetinaNetHead(cfg, in_channels={in_channels}) // RETURNED")
        logger.debug(f"\t// head: {head}")

        logger.debug(f"\t#============================")
        logger.debug(f"\t# 1.2.3 RPN box_coder build")
        logger.debug(f"\t#============================")
        logger.debug(f"\tbox_coder = BoxCoder(weights=(10., 10., 5., 5.)) // CALL\n\t{{")
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        logger.debug(f"\t}}\n\tbox_coder = BoxCoder(weights=(10., 10., 5., 5.)) // RETURNED")
        logger.debug(f"\t// box_coder: {box_coder}")

        logger.debug(f"\t#============================")
        logger.debug(f"\t# 1.2.4 RPN box_selector_test build")
        logger.debug(f"\t#============================")
        logger.debug(f"\tbox_selector_test = make_retinanet_postprocessor(cfg, box_coder) // CALL\n\t{{")
        box_selector_test = make_retinanet_postprocessor(cfg, box_coder)
        logger.debug(f"\t}}\n\tbox_selector_test = make_retinanet_postprocessor(cfg, box_coder) // RETURNED")
        logger.debug(f"\t// box_selector_test: {box_selector_test}")

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        logger.debug(f"\tself.anchor_generator = anchor_generator")
        logger.debug(f"\tself.head = head")
        logger.debug(f"\tself.box_selector_test = box_selector_test")

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

        logger.debug(f"\t\tRetinaNetModule.forward(self, images, features, targets=None) {{ // BEGIN")
        logger.debug(f"\t\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t\t// Params:")
        logger.debug(f"\t\t\t\t// type(images): {type(images)}")
        logger.debug(f"\t\t\t\t// len(images.image_sizes): {len(images.image_sizes)}")
        logger.debug(f"\t\t\t\t// len(images.tensors): {len(images.tensors)}")
        logger.debug(f"\t\t\t\t// len(features)): {len(features)}")
        logger.debug(f"\t\t\t\t// target: {targets}\n")

        logger.debug(f"\t\t\t\t# images info")
        for i, s in enumerate(images.image_sizes):
            logger.debug(f"\t\t\t\t// images.image_sizes[{i}]: {s} # transformed image size (H, W)")

        for i, t in enumerate(images.tensors):
            logger.debug(f"\t\t\t\t// images.tensors[{i}].shape: {t.shape} # zero-padded batch image size (C, H, W)")

        logger.debug(f"\n\t\t\t\t# features info")
        for i, f in enumerate(features):
            if i < 3:
               logger.debug(f"\t\t\t\t// feature[{i}].shape: {f.shape} <== P{i+2} after FPN")
            else:
               logger.debug(f"\t\t\t\t// feature[{i}].shape: {f.shape} <== P{i+3} after FPN")

        logger.debug(f"\n")
        #------------------------
        #  1. head
        #------------------------
        logger.debug(f"\t\t# ===========================================")
        logger.debug(f"\t\t# 2-3-2-1 RPN.Head forward")
        logger.debug(f"\t\t# ===========================================\n")

        logger.debug(f"\t\t// type(self.head): {type(self.head)}")
        logger.debug(f"\t\tbox_cls, box_regression = self.head(features) // CALL\n{{")


        box_cls, box_regression = self.head(features)

        logger.debug(f"\n\t\t}}\nbox_cls, box_regression = self.head(features) // RETURNED")
        logger.debug(f"\t\t// len(box_cls): {len(box_cls)}")
        for i, bc in enumerate(box_cls):
            logger.debug(f"\t\t\t// box_cls[{i}].shape: {bc.shape}")

        logger.debug(f"\t\t// len(box_regression): {len(box_regression)}")
        for i, br in enumerate(box_regression):
            logger.debug(f"\t\t\t// box_regression[{i}].shape: {br.shape}")

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
        logger.debug(f"\n\n")
        logger.debug(f"\t\t# ===========================================")
        logger.debug(f"\t\t# 2-3-2-2 RPN.anchor_generator forward")
        logger.debug(f"\t\t# ===========================================\n")

        logger.debug(f"\t\t// self.anchor_generator: {self.anchor_generator}")
        logger.debug(f"\t\tanchors = self.anchor_generator(images, features) // CALL {{")
        anchors = self.anchor_generator(images, features)
        logger.debug(f"\t\t}}\nanchors = self.anchor_generator(images, features) // RETURNED")
        logger.debug(f"\t\t// anchors: list of list of BoxList")
        logger.debug(f"\t\t// len(anchors):{len(anchors)}")
        logger.debug(f"\t\t// len(anchors[0]):{len(anchors[0])}")
        for i, a in enumerate(anchors[0]):
            logger.debug(f"\t\t// anchors[0][{i}]: {a}")



        #------------------------
        #  3. _forward_test
        #------------------------
        logger.debug(f"\n\n\t\t# ===========================================")
        logger.debug(f"\t\t# 2-3-2-3 RPN._forward_test")
        logger.debug(f"\t\t# ===========================================\n")

        if self.training:
            logger.debug(f"\t\tif self.training: {self.training}")
            logger.debug(f"\t\t\treturn self._forward_train(anchors, box_cls, box_regression, targets)")
            logger.debug(f"\t\t\t}} // END RetinaNetModule.forward(self, images, features, targets=None)")
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            logger.debug(f"\t\tif self.training: {self.training}")

            logger.debug(f"\t\t\t# call paramers info")
            logger.debug(f"\t\t\t# anchors:")
            logger.debug(f"\t\t\t#    from self.anchor_generator(self, x)  # 2-3-2-2")
            logger.debug(f"\t\t\t# box_cls, box_regression:")
            logger.debug(f"\t\t\t#     return value RetinaNetHead.forward(self, x)")
            logger.debug(f"\t\t\t#     called by box_cls, box_regression = self.head(features)  # 2-3-2-1")

            logger.debug(f"\t\t\treturn self._forward_test(anchors, box_cls, box_regression) // CALL")
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_test(self, anchors, box_cls, box_regression):

        logger.debug(f"\n\nRetinaNetModule._forward_test(self, anchors, box_cls, box_regression) {{ // BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t// Params:")    # anchors is list
        logger.debug(f"\t\t// len(anchors): {len(anchors)}")    # anchors is list of list of BoxList
        logger.debug(f"\t\t// anchors: {anchors}")
        logger.debug(f"\t\t// len(box_cls): {len(box_cls)} from RetinaNetHead (RPN.Head cls_towers and cls_logit)")
        logger.debug(f"\t\t// len(box_regression): {len(box_regression)} from RetinaNetHead (RPN.Head bbox_towers and bbox_pred")

        #------------------------
        #  4. box_selector_test
        #------------------------
        logger.debug(f"\t// self.box_selector_test: {self.box_selector_test}")
        logger.debug(f"\tboxes = self.box_selector_test(anchors=>anchors, box_cls=>objectness, box_regression=>box_regression) // CALL\n{{")
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        logger.debug(f"\n")
        logger.debug(f"\t}}\n\tboxes = self.box_selector_test(anchors, box_cls, box_regression) // RETURNED")

        logger.debug(f"\t// len(boxes): {len(boxes)}")
        logger.debug(f"(boxes): {boxes}")
        logger.debug(f"return boxes, {{}} # {{}} is just empty dictionayr")
        logger.debug(f"\n\n}} // RetinaNetModule._forward_test(self, anchors, box_cls, box_regression): END")
        logger.debug(f"}} // END RetinaNetModule.forward(self, images, features, targets=None)")

        # {} is just empty dictionary if self.training == FALSE
        return boxes, {}


def build_retinanet(cfg, in_channels):
    logger.debug(f"\tbuild_retinanet(cfg, in_channels) {{ // BEGIN")
    logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
    logger.debug(f"\t\t// Param:")
    logger.debug(f"\t\t\t// cfg:")
    logger.debug(f"\t\t\t// in_channels: {in_channels}")
    logger.debug(f"\treturn RetinaNetModule(cfg, in_channels) // CALL")

    return RetinaNetModule(cfg, in_channels)
