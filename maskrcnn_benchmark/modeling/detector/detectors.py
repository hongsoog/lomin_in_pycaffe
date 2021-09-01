# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN

# added by LOMIN
from .text_recognizer import TextRecognizer


_DETECTION_META_ARCHITECTURES = {
    "GeneralizedRCNN": GeneralizedRCNN,
    "TextRecognizer": TextRecognizer      # added by LOMIN
}

# This function is the entry function to create a model,
# and it is also the only model creation function
def build_detection_model(cfg):

    # Build a model dictionary, although there is only a pair of key values,
    # but it can be convenient for subsequent expansion
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]

    # The following statement is equivalent to
    # return GeneralizedRCNN(cfg)
    return meta_arch(cfg)
