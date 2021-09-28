# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import inspect
import torch
from torch import nn

# This to_image_list() function is defined in the
# maskrcnn_benchmark/structures/image_list.py file
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn

# added by kimkk for model visualization in tensorboard
# from torch.utils.tensorboard import SummaryWriter
import torchvision

# disable FutureWarning related to NumPy and tensorflow's numpy version
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


# for model debugging log
import logging
from model_log import  logger

# default `log_dir` is "runs"
if logger.level > logging.DEBUG:
    logger.info("SummaryWrite created ... runs/lomin_detect")

    # for tensorboard visualization
    #writer = SummaryWriter('runs/lomin_detect')

# define the specific implementation of the class
class GeneralizedRCNN(nn.Module):
    """
    This class is the common abstraction of all models in maskrcnn_benchmark,
    and currently supports two forms of labels, boxes and masks
    This category mainly contains the following three parts:
    - backbone
    - rpn (optional): region proposal network
    - heads: takes the features and proposals output from the previous network (RPN)
             and calculate detections / masks from them.
    """

    # Initialize the model according to the configuration information
    def __init__(self, cfg):

        logger.debug(f"GeneralizedRCNN.__init__(self, cfg) {{ //BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\tParams:")
        logger.debug(f"\t\t\tcfg:")

        super(GeneralizedRCNN, self).__init__()
        logger.debug(f"\tsuper(GeneralizedRCNN, self).__init__()\n")

        # Create a backbone network based on configuration information
            #logger.debug(f"\tbuild_backbone: {build_backbone}")


        #---------------------------
        # 1. model.backbone build
        #---------------------------
        # backbone = body + fpn
        #---------------------------
        # build_backbone() defined in maskrcnn_benchmark/modelling/backbone/backbone.py
        logger.debug(f"\tself.backbone = build_backbone(cfg) // CALL")

        self.backbone = build_backbone(cfg)

        logger.debug(f"\tself.backbone = build_backbone(cfg) // RETURNED")
        logger.debug(f"\t\tself.backbone: {self.backbone}")

        #---------------------------
        # 2. model.rpn build
        #---------------------------
        # build_rpn() defined in maskrcnn_benchmark/modelling/rpn/rpn.py
        # Create rpn network based on configuration information
        logger.debug(f"\tself.backbone.out_channels: {self.backbone.out_channels}")
        logger.debug(f"\tself.rpn = build_rpn(cfg, self.backbone.out_channels) // CALL")

        self.rpn = build_rpn(cfg, self.backbone.out_channels)

        logger.debug(f"\tself.rpn = build_rpn(cfg, self.backbone.out_channels) // RETURNED")
        logger.debug(f"\tself.rpn: {self.rpn}")


        # Create roi_heads based on configuration information
        # comment out by LOMIN
        #self.roi_heads = build_roi_heads(cfg)

        logger.debug(f"}} END GeneralizedRCNN.__init__(self, cfg)")


    # Define the forward propagation process of the model
    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        # original code removed by LOMIN
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        """
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n\tGeneralizedRCNN.forward(self, images, targets=None) {{ //BEGIN")
            logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}")
            logger.debug(f"\t\tParams:")
            logger.debug(f"\t\t\timages:")
            logger.debug(f"\t\t\t\ttype(images): {type(images)}")
            logger.debug(f"\t\t\ttargets: {targets}")

            logger.debug(f"\tif self.training: {self.training}: ")

        if self.training:
            # prohibit training
            raise

        # Convert the data type of the input image to ImageList
        if logger.level == logging.DEBUG:
            logger.debug(f"\timages = to_image_list(images)")

        images = to_image_list(images)

        #-----------------------------------
        # Backbone: resnet50, fpn
        #-----------------------------------
        # ---------- for debug backbone with tensorboard -------------
        if logger.level == logging.DEBUG:
            logger.debug(f"\timages.image_sizes: {images.image_sizes}")
            logger.debug(f"\timages.tensors.shape: {images.tensors.shape}")

        grid = torchvision.utils.make_grid(images.tensors)
        if logger.level > logging.DEBUG:
            """
            writer.add_image("\tinput_image_to_self.backbone", grid, 0)

            tensors_shape = f"{images.tensors.shape}"
            writer.add_text("\timages.tensors.shape", tensors_shape)

            images_sizes = f"\t{images.image_sizes}"
            writer.add_text("images.images_sizes", images_sizes)

            writer.add_graph(self.backbone, images.tensors, True)
            """
        # ---------- for debug with tensorboard -------------
        if logger.level == logging.DEBUG:
            logger.debug(f"\tmodel.backbone.forward(images.tensors) BEFORE")

        features = self.backbone(images.tensors)

        if logger.level == logging.DEBUG:
            logger.debug(f"\tmodel.backbone.forward(images.tensors) DONE")

        #------------------------------------------------
        # RPN :
        # retinanet used in detection v2 model
        #------------------------------------------------
        # use rpn network to obtain proposals and corresponding loss

        if logger.level > logging.DEBUG:
            pass
            #logger.debug(f"targets: {targets}")
            #logger.debug("writer.add_graph(self.rpn, [images.sizes, images.tensors, features, targets], True)")
            #writer.add_graph(self.rpn, [images.image_sizes, images.tensors, features, targets], True)

        if logger.level == logging.DEBUG:
            logger.debug(f"proposals, proposal_losses = self.rpn(images, features, targets) BEFORE")

        proposals, proposal_losses = self.rpn(images, features, targets)
        #proposals = self.rpn([images.image_sizes, images.tensors, features, targets])

        if logger.level == logging.DEBUG:
            logger.debug(f"proposals, proposal_losses = self.rpn(images, features, targets) DONE")

        # ========= codes removed by LOMIN for preventing training BEGIN
        """
        if self.roi_heads: 
            # how to calculate the output result if roi_heads is not None
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
         
        if self.training: 
             # In training mode, output loss value
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

        # If it is not in training mode, output the prediction result of the model.
        return result
        """
        # ========= codes removed by LOMIN for preventing training END


        if logger.level == logging.DEBUG:
            logger.debug(f"x = features");
            logger.debug(f"result = proposals");

        x = features    # what for this code?
        result = proposals

        # If it is not in training mode, output the prediction result of the model.
        if logger.level == logging.DEBUG:
            logger.debug(f"return result");
            logger.debug(f"}} // END GeneralizedRCNN.forward(self, images, targets=None)")
        return result
