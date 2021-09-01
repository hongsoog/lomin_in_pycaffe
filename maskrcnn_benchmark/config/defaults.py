# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import sys

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.RETINANET_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN" # GeneralizedRCNN | TextRecognizer


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.RESIZE_MODE = 'keep_ratio'
_C.INPUT.FIXED_SIZE = (-1, -1)
_C.INPUT.MIN_SIZE_TEST = 800
_C.INPUT.MAX_SIZE_TEST = 1333
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
_C.INPUT.PIXEL_STD = [1., 1., 1.]
_C.INPUT.TO_BGR255 = True
_C.INPUT.TO_N1P1 = False
_C.INPUT.TARGET_INTERPOLATION = 'bilinear'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.SIZE_DIVISIBILITY = 0

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False

# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
_C.MODEL.GROUP_NORM.EPSILON = 1e-5

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RPN.STRADDLE_THRESH = 0
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
_C.MODEL.RPN.NMS_THRESH = 0.7
_C.MODEL.RPN.MIN_SIZE = 0
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
_C.MODEL.RPN.FPN_POST_NMS_PER_BATCH = True
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()
_C.MODEL.RESNETS.NUM_GROUPS = 1
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"
_C.MODEL.RESNETS.RES5_DILATION = 1
_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1

# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()
_C.MODEL.RETINANET.NUM_CLASSES = 81
_C.MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.RETINANET.STRADDLE_THRESH = 0
_C.MODEL.RETINANET.OCTAVE = 2.0
_C.MODEL.RETINANET.SCALES_PER_OCTAVE = 3
_C.MODEL.RETINANET.USE_C5 = True
_C.MODEL.RETINANET.NUM_CONVS = 4
_C.MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0
_C.MODEL.RETINANET.BBOX_REG_BETA = 0.11
_C.MODEL.RETINANET.PRE_NMS_TOP_N = 1000
_C.MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5
_C.MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4
_C.MODEL.RETINANET.LOSS_ALPHA = 0.25
_C.MODEL.RETINANET.LOSS_GAMMA = 2.0
_C.MODEL.RETINANET.PRIOR_PROB = 0.01
_C.MODEL.RETINANET.INFERENCE_TH = 0.05
_C.MODEL.RETINANET.NMS_TH = 0.4

# ---------------------------------------------------------------------------- #
# TIMM (pytorch-image-models) Option
# ---------------------------------------------------------------------------- #
_C.MODEL.TIMM = CN()
_C.MODEL.TIMM.USE_PRETRAINED = True
_C.MODEL.TIMM.BACKBONE_OUT_CHANNELS = 256
_C.MODEL.TIMM.OUT_INDICES = (2, 3, 4)

# ---------------------------------------------------------------------------- #
# TextRecognizer options
# ---------------------------------------------------------------------------- #
_C.MODEL.RECOGNITION = False
_C.MODEL.TEXT_RECOGNIZER = CN()
_C.MODEL.TEXT_RECOGNIZER.CHARACTER = ("num", "eng_cap", "eng_low", "kor_2350")
_C.MODEL.TEXT_RECOGNIZER.BATCH_MAX_LENGTH = 25
_C.MODEL.TEXT_RECOGNIZER.CNN_CHANNELS = 512
_C.MODEL.TEXT_RECOGNIZER.CNN_NUM_POOLING = 4
_C.MODEL.TEXT_RECOGNIZER.USE_PROJECTION = False
_C.MODEL.TEXT_RECOGNIZER.OUTCONV_KS = (1, 1)
_C.MODEL.TEXT_RECOGNIZER.TRANSFORMER_FSIZE = 512
_C.MODEL.TEXT_RECOGNIZER.TRANSFORMER_MODULE_NO = 6
_C.MODEL.TEXT_RECOGNIZER.TRANSFORMER_ENCODER_MODULE_NO = -1
_C.MODEL.TEXT_RECOGNIZER.TRANSFORMER_DECODER_MODULE_NO = -1
_C.MODEL.TEXT_RECOGNIZER.TRANSFORMER_NO_RECURRENT_PATH = False

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.DETECTIONS_PER_IMG = 100
_C.TEST.SCORE_THRESHOLD = 0.5
_C.TEST.NMS_THRESH = 0.5
