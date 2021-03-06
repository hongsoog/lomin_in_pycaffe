# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
import logging
import inspect
import pprint
import numpy as np
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.utils.registry import Registry

# for model debugging log
from model_log import logger

# ---------------------------------
# ResNet stage specification
# ---------------------------------
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the feature map from this stage
    ],
)

# ==================================
# ResNet models list
# ==================================

# ---------------------------------
# 1. ResNet-50 (including all stages)
# ---------------------------------
# The number of conv layers in the 2~5 stage: 3, 4, 6, 3
# return feature map from last stage (5-stage)
# ---------------------------------
ResNet50StagesTo5 = tuple(
    # The element type inside the tuple is StageSpec
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)

# ---------------------------------
# 2. ResNet-50-C4 up to stage 4 (excluding stage 5)
# ---------------------------------
# The number of conv layers in the 2~4 stage: 3, 4, 6
# only use the feature map output from the 4th stage
# ---------------------------------
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)

# ---------------------------------
# 3. ResNet-101 (including all stages)
# ---------------------------------
# The number of conv layers in the 2~5 stage: 3, 4, 23, 3
# return feature map from last stage (5-stage)
# ---------------------------------
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)

# ---------------------------------
# 4. ResNet-101-C4 up to stage 4 (excludes stage 5)
# ---------------------------------
# The number of conv layers in the 2~5 stage: 3, 4, 23, 3
# only use the feature map output from the 4th stage
# ---------------------------------
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)

# ---------------------------------
# 5. ResNet-50-FPN (including all stages)
# ---------------------------------
# The number of conv layers in the 2~4 stages are: 3, 4, 6, 3
# As FPN needs to use the feature map output by each stage,
# the return_features parameter is True
# ---------------------------------
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)

# ---------------------------------
# 5. ResNet-101-FPN (including all stages)
# ---------------------------------
# The number of convolutional layers are: 3, 4, 23, 3
# As FPN needs to use the feature map output by each stage,
# the return_features parameter is True
# ---------------------------------
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)

# ---------------------------------
# 7. ResNet-152-FPN (including all stages)
# ---------------------------------
# The number of convolutional layers are: 3, 8, 36, 3
# As FPN needs to use the feature map output by each stage,
# the return_features parameter is True
# ---------------------------------
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)

"""
ResNet Models Summary
+-----------------------+---------+--------------------------------------------------+-------------------------------+
|                       | Num. of |          Num. of Residual Blocks                 |                               |
|   Model ID            | stage   +---------+---------+---------+----------+---------+          Remarks              |
|                       |         | stage 1 | stage 2 | stage 3 |  stage 4 | stage 5 |                               |
+-----------------------+---------+---------+---------+---------+----------+---------+-------------------------------+
| ResNet50StagesTo5     |   5     |         |    3    |    4    |     6    |    3    | return stage 5 feature map    |
+-----------------------+---------+---------+---------+---------+----------+---------+-------------------------------+
| ResNet50StagesTo4     |   4     |         |    3    |    4    |     6    |    X    | retrun stage 4 feature map    |
+-----------------------+---------+---------+---------+---------+----------+---------+-------------------------------+
| ResNet101StagesTo5    |   5     |         |    3    |    4    |    23    |    3    | return stage 5 feature map    |
+-----------------------+---------+---------+---------+---------+----------+---------+-------------------------------+
| ResNet1010StagesTo4   |   4     |         |    3    |    4    |    23    |    X    | return stage 4 feature map    |
+-----------------------+---------+---------+---------+---------+----------+---------+-------------------------------+
| ResNet50FPNStagesTo5  |   5     |         |    3    |    4    |     6    |    3    | return stage 2-5 feature maps |
+-----------------------+---------+---------+---------+---------+----------+---------+-------------------------------+
| ResNet101FPNStagesTo5 |   5     |         |    3    |    4    |    23    |    3    | return stage 2-5 feature maps |
+-----------------------+---------+---------+---------+---------+----------+---------+-------------------------------+
| ResNet152FPNStagesTo5 |   5     |         |    3    |    8    |    36    |    3    | return stage 2-5 feature maps |
+-----------------------+---------+---------+---------+---------+----------+---------+-------------------------------+
"""


# ---------------------------------
# ResNet class
# ---------------------------------
class ResNet(nn.Module):


    def __init__(self, cfg):

        logger.debug(f"\n\tResnet.__init__(self, cfg) {{ //BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t// Params:")
        logger.debug(f"\t\t\t// cfg:\n")

        logger.debug(f"\t\tsuper(ResNet, self).__init__()\n")
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate the string names in the conf file into corresponding specific implementations.
        # The following three use the corresponding registered modules, which are defined at the end of the file
        logger.debug(f"\t\t// _STEM_MODULES: {_STEM_MODULES}\n")
        logger.debug(f"\t\t// _cfg.MODEL.RESNETS.STEM_FUNC: {cfg.MODEL.RESNETS.STEM_FUNC}")

        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]

        logger.debug(f"\t\tstem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]")
        logger.debug(f"\t\t// stem_module: {stem_module}\n")

        # resnet conv2_x~conv5_x implementation
        # ex. cfg.MODEL.CONV_BODY="R-50-FPN"
        #logger.debug(f"\t\t_STAGE_SPECS: {_STAGE_SPECS}")
        logger.debug(f"\t\t// cfg.MODEL.BACKBONE.CONV_BODY: {cfg.MODEL.BACKBONE.CONV_BODY}")
        logger.debug(f"\t\tstage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY={cfg.MODEL.BACKBONE.CONV_BODY}]")

        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]

        logger.debug(f"\t\t// stage_specs: {stage_specs}\n")

        # residual transformation function
        # ex. cfg.MODEL.RESNETS.TRANS_FUNC="BottleneckWithFixedBatchNorm"
        logger.debug(f"\t\t// _TRANSFORMATION_MODULES: {_TRANSFORMATION_MODULES}")
        logger.debug(f"\t\t// cfg.MODEL.RESNETS.TRANS_FUNC: {cfg.MODEL.RESNETS.TRANS_FUNC}")

        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]
        logger.debug(f"\t\ttransformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC={cfg.MODEL.RESNETS.TRANS_FUNC}]")
        logger.debug(f"\t\t// transformation_module: {transformation_module}")

        # After obtaining the implementation of each of the above components,
        # you can use these implementations to build the model

        # construct the stem module
        # that is, stage 1 of resnet : conv1 -> bn1 -> relu -> max_pool2d
        logger.debug(f"\t\tself.stem = stem_module(cfg) // CALL")
        logger.debug(f"\t\t{{")

        self.stem = stem_module(cfg)

        logger.debug(f"\n\t\t}}")
        logger.debug(f"\t\tself.stem = stem_module(cfg) // RETURNED")
        logger.debug(f"\t\t// self.stem: {self.stem}")

        # Construct the specified ResNet stages
        # obtain the corresponding information to construct the convolutional layer of other stages of ResNet

        # ResNet when num_groups=1, ResNeXt when >1
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS

        logger.debug(f"\t\tnum_groups = cfg.MODEL.RESNETS.NUM_GROUPS")
        logger.debug(f"\t\t// num_groups.stem: {num_groups}")

        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP

        logger.debug(f"\t\twidth_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP")
        logger.debug(f"\t\t// width_per_group: {width_per_group}")


        # in_channels refers to the number of channels of the feature map
        # when inputting to the stage 2 and later, that is the number of stem output channels, the default is 64
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        logger.debug(f"\t\tin_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS")
        logger.debug(f"\t\t// in_channels: {in_channels}")

        # The number of channels of the special map input in the stage 2
        stage2_bottleneck_channels = num_groups * width_per_group

        logger.debug(f"\t\tstage2_bottleneck_channels = num_groups * width_per_group")
        logger.debug(f"\t\t// stage2_bottleneck_channels: {stage2_bottleneck_channels}")

        # The num. of channels of output of the stage 2
        # ResNet series standard model can judge the number of subsequent channels
        # from the output channel number of the stage 2 of ResNet
        #    The default is 256, then the follow-ups are 512, 1024, 2048 respectively,
        #    if it is 64, the follow-ups are 128, 256, 512 respectively
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS

        logger.debug(f"\t\tstage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS")
        logger.debug(f"\t\tstage2_out_channels: {stage2_out_channels}")

        # Create an empty stages list and corresponding feature map dictionary
        self.stages = []
        self.return_features = {}

        logger.debug(f"\t\tself.stages = []")
        logger.debug(f"\t\tself.return_features = {{}}")

        # For the definition of stage_specs,
        # refer to ResNet stage specification in this file
        logger.debug(f"\t\tfor stage_spec in stage_specs {{\n")

        total_iter = len(stage_specs)
        for i, stage_spec in enumerate(stage_specs):
            logger.debug(f"\n\t\t\t{{")
            logger.debug(f"\t\t\t# ---------------------------------------------------------------")
            logger.debug(f"\t\t\t# iteration {i+1}/{total_iter}")
            logger.debug(f"\t\t\t#  {i+1}-th stage_spec: {stage_spec}")
            logger.debug(f"\t\t\t# ---------------------------------------------------------------")

            name = "layer" + str(stage_spec.index)

            logger.debug(f'\t\t\tname = "layer" + str(stage_spec.index)')
            logger.debug(f"\t\t\t// name: {name}\n")

            # Calculate the number of output channels for each stage,
            # each time through a stage, the number of channels will be doubled
            stage2_relative_factor = 2 ** (stage_spec.index - 1)

            logger.debug(f"\t\t\tstage2_relative_factor = 2 ** (stage_spec.index - 1)")
            logger.debug(f"\t\t\t// stage2_relative_factor: {stage2_relative_factor}\n")

            # Calculate the number of input channels
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

            logger.debug(f"\t\t\tbottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor")
            logger.debug(f"\t\t\t// bottlenec_channels: {bottleneck_channels}\n")

            # Calculate the number of output channels
            out_channels = stage2_out_channels * stage2_relative_factor

            logger.debug(f"\t\t\tout_channels = stage2_out_channels * stage2_relative_factor")
            logger.debug(f"\t\t\t// out_channels: {out_channels}\n")

            #
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]

            logger.debug(f"\t\t\tstage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]")
            logger.debug(f"\t\t\t// stage_with_dcn: {stage_with_dcn}\n")

            # When all the required parameters are obtained, call the `_make_stage` function of this file,
            # This function can create a module corresponding to the stage according to the parameters passed in.
            logger.debug(f"\t\t\tmodule = _make_stage(")
            logger.debug(f"\t\t\t\ttransformation_module = {transformation_module},")
            logger.debug(f"\t\t\t\tin_channels = {in_channels},")
            logger.debug(f"\t\t\t\tbottleneck_channels = {bottleneck_channels},")
            logger.debug(f"\t\t\t\tout_channels = {out_channels},")
            logger.debug(f"\t\t\t\tstage_spec.block_count = {stage_spec.block_count},")
            logger.debug(f"\t\t\t\tnum_groups = {num_groups},")
            logger.debug(f"\t\t\t\tcfg.MODEL.RESNETS.STRIDE_IN_1X1 : {cfg.MODEL.RESNETS.STRIDE_IN_1X1},")
            logger.debug(f"\t\t\t\tfirst_stride=int(stage_spec.index > 1) + 1: {int(stage_spec.index > 1) +1},")
            logger.debug(f"\t\t\t\tdcn_config={{")
            logger.debug(f"\t\t\t\t\t'stage_with_dcn': {stage_with_dcn},")
            logger.debug(f"\t\t\t\t\t'with_modulated_dcn': {cfg.MODEL.RESNETS.WITH_MODULATED_DCN},")
            logger.debug(f"\t\t\t\t\t'deformable_groups': {cfg.MODEL.RESNETS.DEFORMABLE_GROUPS},")
            logger.debug(f"\t\t\t\t\t}}")
            logger.debug(f"\t\t\t\t) // CALL")

            logger.debug(f"\t\t\t\t{{")
            # Note that that it is not a model but a module (nn.Module)
            module = _make_stage(
                transformation_module,
                in_channels,  # num. of input ch
                bottleneck_channels,  # num. of compressed ch.
                out_channels,  # num. of output ch
                stage_spec.block_count,  # num. of cov layers (residual blocks) in the current stage
                num_groups,  # num_groups=1 means Resnet and num_groups>1 means ResNetX
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,

                # For stage 3, 4 and 5, use stride=2
                # to downsize at the beginning of residual block
                first_stride=int(stage_spec.index > 1) + 1,

                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
                }
            )

            logger.debug(f"\t\t\t\t}}")

            logger.debug(f"\t\t\tmodule = _make_stage(")
            logger.debug(f"\t\t\t\ttransformation_module = {transformation_module},")
            logger.debug(f"\t\t\t\tin_channels = {in_channels},")
            logger.debug(f"\t\t\t\tbottleneck_channels = {bottleneck_channels},")
            logger.debug(f"\t\t\t\tout_channels = {out_channels},")
            logger.debug(f"\t\t\t\tstage_spec.block_count = {stage_spec.block_count},")
            logger.debug(f"\t\t\t\tnum_groups = {num_groups},")
            logger.debug(f"\t\t\t\tcfg.MODEL.RESNETS.STRIDE_IN_1X1 : {cfg.MODEL.RESNETS.STRIDE_IN_1X1},")
            logger.debug(f"\t\t\t\tfirst_stride=int(stage_spec.index > 1) + 1: {int(stage_spec.index > 1) +1},")
            logger.debug(f"\t\t\t\tdcn_config={{")
            logger.debug(f"\t\t\t\t\t'stage_with_dcn': {stage_with_dcn},")
            logger.debug(f"\t\t\t\t\t'with_modulated_dcn': {cfg.MODEL.RESNETS.WITH_MODULATED_DCN},")
            logger.debug(f"\t\t\t\t\t'deformable_groups': {cfg.MODEL.RESNETS.DEFORMABLE_GROUPS},")
            logger.debug(f"\t\t\t\t\t}}")
            logger.debug(f"\t\t\t\t) // RETURNED\n")

            # The num. of output ch of the stage i (current stage) is
            # the num. of input ch. of the stage i+1  (next stage)
            in_channels = out_channels
            logger.debug(f"\t\t\tin_channels = out_channels")
            logger.debug(f"\t\t\t// in_channels: {in_channels}\n")

            # Add the current stage module to the model
            self.add_module(name, module)

            #logger.debug(f"\t\t\tself.add_module(name={name}, module={module})")
            logger.debug(f"\t\t\tself.add_module(name={name}, module)\n")

            # Add the name of the stage to the list
            self.stages.append(name)

            logger.debug(f"\t\t\tself.stages.append(name={name})\n")

            # Add the boolean value of the stage to the dictionary
            # it indicates that whether feature map of current stage returns or not
            self.return_features[name] = stage_spec.return_features

            logger.debug(f"\t\t\t// name: {name}")
            logger.debug(f"\t\t\t// stage_spec.return_features: {stage_spec.return_features}")
            logger.debug(f"\t\t\tself.return_features[name] = stage_spec.return_features\n")

            logger.debug(f"\t\t\t}}  // END of iteration {i+1}/{total_iter}\n")
        logger.debug(f"}} // END for stage_spec in stage_specs:\n")

        # Selectively freeze (requires_grad=False) layer in the backbone
        # Selectively freeze certain layers according to the parameters
        # of the conf file (requires_grad=False)
        logger.debug(f"\t\t\t// cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT: {cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT})")
        logger.debug(f"\t\t\tself._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)")

        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

        logger.debug(f"}} // END Resnet.__init__(self, cfg)\n")

    # Freeze the parameter updates of certain layers according to the given parameters
    def _freeze_backbone(self, freeze_at):
        logger.debug(f"\n\tResnet.__freeze_backbone(self, freeze_at) {{ // BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t// Params:")
        logger.debug(f"\t\t\t// freeze_at: {freeze_at}")

        if freeze_at < 0:
            return

        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 of ResNet is stem
            else:
                m = getattr(self, "layer" + str(stage_index))

            # set all parameters in m to non-updated state
            for p in m.parameters():
                p.requires_grad = False

        logger.debug(f"\t}} // END Resnet.__freeze_backbone(self, freeze_at)\n")

    # Define the forward propagation process of ResNet
    def forward(self, x):
        logger.debug(f"\n\n")
        logger.debug(f"\t\t# =================================")
        logger.debug(f"\t\t# 2-3-1-1 ResNet Forward")
        logger.debug(f"\t\t# =================================\n")


        logger.debug(f"\tResnet.forward(self, x) {{ //BEGIN")
        logger.debug(f"\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\tParam")
        logger.debug(f"\t\t\tx.shape={x.shape}\n")

        outputs = []

        # First go through the stem(layer 0)

        logger.debug(f"\t\t# =================================")
        logger.debug(f"\t\t# 2-3-1-1-1 stem (layer0) forward")
        logger.debug(f"\t\t# =================================\n\n")
        logger.debug(f"\t\tx = self.stem(x) {{ // CALL\n")

        x = self.stem(x)

        logger.debug(f"\n\t\t}}")
        logger.debug(f"\n\t\tx = self.stem(x) // RETURNED")
        logger.debug(f"\t\t// x.shape: {x.shape}\n")
        file_path = f"./npy_save/stem_output"
        arr = x.cpu().numpy()
        np.save(file_path, arr)
        logger.debug(f"\t\tstem output of shape {arr.shape} saved into {file_path}.npy\n\n")

        # Then calculate the results of layer 1 ~ 4 in turn
        #logger.debug(f"\tfor stage_name in self.stages")
        logger.debug(f"\t\tfor stage_name in self.stages:\n\t\t{{")

        for stage_name in self.stages:

            # -------------------------------------------------
            # Notes
            # ResNet 50 with 5 stages
            # -------------------------------------------------
            # stem layer
            # layer1  ==> C1 (not used in FPN)
            # layer2  ==> C2
            # layer3  ==> C3
            # layer4  ==> C4
            # -------------------------------------------------

            stage_num = f"{int(stage_name[-1])}"

            # run layer 2-5
            logger.debug(f"\t\t\t# =================================")
            logger.debug(f"\t\t\t# 2-3-1-1-{stage_num} forward")
            logger.debug(f"\t\t\t# =================================")
            logger.debug(f"\t\t\t{{ // BEGIN of iteration for {stage_num} for {stage_name}\n")

            logger.debug(f"\t\t\tx = getattr(self, stage_name)(x) {{ // CALL\n")
            x = getattr(self, stage_name)(x)
            logger.debug(f"\n\t\t\t}} // x = getattr(self, stage_name)(x) RETURNED\n")

            logger.debug(f"\t\t\t\t// output shape of {stage_name}: {x.shape}\n")

            # Save all the calculation results of stage 1 ~ 4 (that is, the feature map) in the form of a list
            logger.debug(f"\t\t\t# Save all the calculation results of stage 1 ~ 4 (that is, the feature map) in the form of a list")
            if self.return_features[stage_name]:
                logger.debug(f"\t\t\tif self.return_features[stage_name]:")
                outputs.append(x)

                logger.debug(f"\t\t\t\toutputs.append(x)")
                logger.debug(f"\t\t\t\t// stage_name: {stage_name}")
                logger.debug(f"\t\t\t\t// x.shape: {x.shape}\n")

                file_path = f"./npy_save/C{stage_num}"
                arr = x.cpu().numpy()
                np.save(file_path, arr)
                logger.debug(f"\t\t\t\t#{stage_name} output of shape {arr.shape} saved into {file_path}.npy\n\n")

            logger.debug(f"\t\t\t}} // END of iteration for {stage_name}\n")

        logger.debug(f"\n\t\t}} // END for stage_name in self.stages\n")

        # Return the results, outputs are in the form of a list, and the elements are the feature maps of each stage,
        # which happen to be the input of FPN

        logger.debug(f"\t\t\t# -------------------------------")
        logger.debug(f"\t\t\t# return value (outputs) info")
        logger.debug(f"\t\t\t# fed into FPN.forward()")
        logger.debug(f"\t\t\t# -------------------------------")
        for idx, output in enumerate(outputs):
            logger.debug(f"\t\t\toutput of layer{idx+1} is C{idx+1} of shape: {output.shape}")

        logger.debug(f"\n\t\treturn outputs")
        logger.debug(f"\n\t}} // END Resnet.forward(self, x)")
        return outputs


# ---------------------------------
# ResNetHead class
# ---------------------------------
class ResNetHead(nn.Module):
    def __init__(self, block_module, stages, num_groups=1, width_per_group=64,
                 stride_in_1x1=True, stride_init=None, res2_out_channels=256,
                 dilation=1, dcn_config={}):

        logger.debug(f"\n\tResNetHead.init__() {{ //BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t// Params:")

        super(ResNetHead, self).__init__()

        # calculate the multiples of the num. of ch at different stages
        # relative to stage2
        stage2_relative_factor = 2 ** (stages[0].index - 1)

        # calculate num. of ch for stage 2 bottleneck
        stage2_bottleneck_channels = num_groups * width_per_group

        # calculate num. of output ch
        out_channels = res2_out_channels * stage2_relative_factor

        # calculate num. of input  ch
        in_channels = out_channels // 2

        # calculate num. of ch for bottleneck
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        # get the corresponding block_module with name
        # currently, only _TRANSFORMATION_MODULES contains the "BottleneckWithFixedBatchNorm" module
        block_module = _TRANSFORMATION_MODULES[block_module]

        # create empty list for stages
        self.stages = []

        # initialize stride
        stride = stride_init

        # iterate over stages
        for stage in stages:

            name = "layer" + str(stage.index)

            if not stride:
                # for stage 3, 4,  and 5, use stride=2
                # to downsize at the beginning of residual block
                stride = int(stage.index > 1) + 1

            module = _make_stage(
                block_module,
                in_channels,  # num. of input ch
                bottleneck_channels,  # num. of bottleneck ch
                out_channels,  # num. of output ch
                stage.block_count,  #
                num_groups,  # num_groups=1 means Resnet and num_groups>1 menas ResNetX
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)

        self.out_channels = out_channels
        logger.debug(f"\n\t}} // END ResNetHead.init__()")

    def forward(self, x):
        logger.debug(f"\n\tResNetHead.forward(self.x) {{ //BEGIN")
        logger.debug(f"\t\t// Params:")
        logger.debug(f"\t\t\tx: {x}")

        for stage in self.stages:
            x = getattr(self, stage)(x)

        logger.debug(f"\n\treturn x)")
        logger.debug(f"\n\t}} // END ResNetHead.forward(self.x)")
        return x


def _make_stage(transformation_module, in_channels, bottleneck_channels, out_channels, block_count, num_groups,
                stride_in_1x1, first_stride, dilation=1, dcn_config={}):
    logger.debug(f"\n\t_make_stage() {{ //BEGIN")
    logger.debug(f"\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
    logger.debug(f"\t\t// Params:")
    logger.debug(f"\t\t\t// transformation_module: {transformation_module}")
    logger.debug(f"\t\t\t// in_channels: {in_channels}")
    logger.debug(f"\t\t\t// bottleneck_channels: {bottleneck_channels}")
    logger.debug(f"\t\t\t// out_channels: {out_channels}")
    logger.debug(f"\t\t\t// block_count: {block_count}")
    logger.debug(f"\t\t\t// num_groups: {block_count}")
    logger.debug(f"\t\t\t// stride_in_1x1: {stride_in_1x1}")
    logger.debug(f"\t\t\t// first_stride: {first_stride}")
    logger.debug(f"\t\t\t// dilation: {dilation}")
    logger.debug(f"\t\t\t// dcn_config: {dcn_config}\n")

    blocks = []
    logger.debug(f"\t\tblocks = []\n")

    stride = first_stride
    logger.debug(f"\t\tstride = first_stride\n")

    logger.debug(f"\t\tfor _ in range(block_count):  {{")

    i =1
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
        )

        logger.debug(f"\t\t\t{{")
        logger.debug(f"\t\t\t# --------------")
        logger.debug(f"\t\t\t# blocks.append iteration {i}/{block_count}")
        logger.debug(f"\t\t\t# --------------")
        logger.debug(f"\t\t\tblocks.append(")
        logger.debug(f"\t\t\t    transformation_module(")
        logger.debug(f"\t\t\t        in_channels={in_channels},")
        logger.debug(f"\t\t\t        bottleneck_channels={bottleneck_channels},")
        logger.debug(f"\t\t\t        out_channels={out_channels},")
        logger.debug(f"\t\t\t        num_groups={num_groups},")
        logger.debug(f"\t\t\t        stride_in_1x1={stride_in_1x1},")
        logger.debug(f"\t\t\t        stride={stride},")
        logger.debug(f"\t\t\t        dilation={dilation},")
        logger.debug(f"\t\t\t        dcn_config={dcn_config}")
        logger.debug(f"\t\t\t   )")
        logger.debug(f"\t\t\t)\n")

        logger.debug(f"\t\t\tblocks[-1]:{blocks[-1]}")

        stride = 1
        logger.debug(f"\t\t\tstride = 1")

        in_channels = out_channels
        logger.debug(f"\t\t\tin_channels = out_channels")

        i = i +1

        logger.debug(f"\n\t\t\t}}")

    logger.debug(f"\t\t}}// END for _ in range(block_count):\n")

    logger.debug(f"\t\treturn nn.Sequential(*blocks)\n")
    logger.debug(f"\n\t}} // END _make_stage()")
    return nn.Sequential(*blocks)


# ------------------------------------------
# Bottleneck class - inherits nn.Module
# ------------------------------------------
class Bottleneck(nn.Module):
    def __init__(
            self,
            in_channels,          # bottleneck input channels
            bottleneck_channels,  # bottleneck output channels
            out_channels,         # output channels of the current stage
            num_groups,
            stride_in_1x1,
            stride,
            dilation,
            norm_func,
            dcn_config
    ):
        #logger.debug(f"\n\tBottleneck.__init__() {{ //BEGIN")
        #logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        #logger.debug(f"\t\t// Params:")

        super(Bottleneck, self).__init__()

        """
        downsample
        When the input and output channels of the bottleneck are not equal, a certain strategy needs to be adopted.
        In the original paper, there are three strategies A, B, and C. Here we use strategy B (also recommended 
        by the original paper)
        That is, **projection shortcuts** are used only when the number of input and output channels are not equal,
         - the parameter matrix mapping is used to make the input and output channels equal
        """
        self.downsample = None

        # When the number of input and output channels is different,
        # add an additional 1??1 conv layer to map the number of input channels to the number of output channels
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                # Backstreet a BN layer with fixed parameters
                norm_func(out_channels),
            )
            for modules in [self.downsample, ]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1  # reset to be 1

        """
        The original MSRA ResNet models have stride in the first 1x1 conv
        The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        stride in the 3x3 conv
        """
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        # create the first conv layer of bottleneck
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )

        # followed by the first BN layer with fixed parameters
        self.bn1 = norm_func(bottleneck_channels)

        # create the second conv layer of bottleneck
        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation
        )
        nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        # followed by the second BN layer with fixed parameters
        self.bn2 = norm_func(bottleneck_channels)

        # create the third conv layer of bottleneck
        # padding defaults to 1
        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        # followed by the third BN layer with fixed parameters
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)

        #logger.debug(f"\n\t}} // END Bottelneck.__init__()")

    def forward(self, x):
        """
        Performing a forward is equivalent to performing a bottleneck,
        By default, there are three convolutional layers, one identity connection, and BN and relu activations
        after each convolutional layer
        Note that the last activation function should be placed after the identity connection
        """

        logger.debug(f"\n\tBottleneck.__forward__(self.x) {{ //BEGIN")
        logger.debug(f"\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\tParams:")
        logger.debug(f"\t\t\tx.shape : {x.shape}\n")
        logger.debug(f"\t\t\tself : {self}\n")

        # Identity connection, directly make the residual equal to x
        logger.debug(f"\t\t// # Identity connection, directly make the residual equal to x")

        identity = x
        logger.debug(f"\t\tidentity = x\n")

        # conv1, bn1, relu (in place relu)
        logger.debug(f"\t\t// # conv1, bn1, relu (in place relu)")
        logger.debug(f"\t\t// self.conv1: {self.conv1}")
        logger.debug(f"\t\t// self.bn1: {self.bn1}\n")

        out = self.conv1(x)
        logger.debug(f"\t\tout = self.conv1(x)")
        logger.debug(f"\t\t// out.shape: {out.shape}\n")

        out = self.bn1(out)
        logger.debug(f"\t\tout = self.bn1(out)")
        logger.debug(f"\t\t// out.shape: {out.shape}\n")

        out = F.relu_(out)
        logger.debug(f"\t\tout = F.relu_(out)")
        logger.debug(f"\t\t// out.shape: {out.shape}\n\n")

        # conv2, bn2, relu (in place relu)
        logger.debug(f"\t\t// # conv2, bn2, relu (in place relu)")
        logger.debug(f"\t\t// self.conv2: {self.conv2}")
        logger.debug(f"\t\t// self.bn2: {self.bn2}\n")

        out = self.conv2(out)
        logger.debug(f"\t\tout = self.conv2(x)")
        logger.debug(f"\t\t// out.shape: {out.shape}\n")

        out = self.bn2(out)
        logger.debug(f"\t\tout = self.bn2(out)")
        logger.debug(f"\t\t// out.shape: {out.shape}\n")

        out = F.relu_(out)
        logger.debug(f"\t\tout = F.relu_(out)")
        logger.debug(f"\t\t// out.shape: {out.shape}\n\n")

        # conv3, bn3, not relu here
        logger.debug(f"\t\t// # conv3, bn3, not relu here")
        logger.debug(f"\t\t// self.conv3: {self.conv3}")
        logger.debug(f"\t\t// self.bn3: {self.bn3}\n")

        out = self.conv3(out)
        logger.debug(f"\t\tout = self.conv3(x)")
        logger.debug(f"\t\t// out.shape: {out.shape}\n")

        out = self.bn3(out)
        logger.debug(f"\t\tout = self.bn3(out)")
        logger.debug(f"\t\t// out.shape: {out.shape}\n\n")

        # If the number of input and output channels are different,
        # they need to be mapped to make them the same.
        logger.debug(f"\t\t// # If the number of input and output channels are different,")
        logger.debug(f"\t\t// # they need to be mapped to make them the same.")

        logger.debug(f"\t\t// self.downsample: {self.downsample}")
        if self.downsample is not None:
            logger.debug(f"\t\tif self.downsample is not None:")

            identity = self.downsample(x)
            logger.debug(f"\t\t\tidentity = self.downsample(x)")
            logger.debug(f"\t\t\t// identity.shape: {identity.shape}\n")


        out += identity   # H = F + x in paper
        logger.debug(f"\t\tout += identity   # H = F + x in paper")
        logger.debug(f"\t\t// out.shape: {out.shape}\n\n")

        # the third (final) relu ( in place relu)
        logger.debug(f"\t\t// # the third (final) relu ( in place relu)")
        out = F.relu_(out)
        logger.debug(f"\t\tout = F.relu_(out)")
        logger.debug(f"\t\t// out.shape: {out.shape}\n")

        logger.debug(f"\t\treturn out\n")

        logger.debug(f"\n\t}} // END Bottelneck.__forward__()")
        # return the convolution result with residual term
        return out


# ------------------------------------------
# BottleneckWithFixedBatchNorm - inherits Bottleneck
# ------------------------------------------
class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
            self,
            in_channels,
            bottleneck_channels,
            out_channels,
            num_groups=1,
            stride_in_1x1=True,
            stride=1,
            dilation=1,
            dcn_config={}
    ):
        #logger.debug(f"\n\tBottleneckWithFixedBatchNorm.__init__() {{ //BEGIN")
        #logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")

        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config
        )

        #logger.debug(f"\n\t}} // END BottelneckWithFixedBatchNorm.__init__()")


# ------------------------------------------
# BaseStem - inherits nn.Module
# ------------------------------------------
class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):

        logger.debug(f"\n\tBaseStem.__init__() {{ //BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")

        super(BaseStem, self).__init__()

        # for ResNet50, num of output ch in stage 1 (stem) is 64
        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        # 7x7 conv layer : in: 3 ch, out: 64 ch
        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )

        # FronzenBatchNorm2d
        self.bn1 = norm_func(out_channels)

        # init weight parameters in conv1 layer with He (Kaiming He)
        for l in [self.conv1, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)

        logger.debug(f"\n\t}} // END BaseStem.__init__()")

    # define the forward propagation process
    # Note:
    # the layers such as ReLu and MaxPool2d, which does not contain learnable parameters,
    # are place in the forward() function and use functions defined in torch.nn.functional
    def forward(self, x):

        logger.debug(f"\n\tBaseStem.forward(self, x) {{ //BEGIN")
        logger.debug(f"\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t// Params:")
        logger.debug(f"\t\t\t> x.shape: {x.shape}\n")

        x = self.conv1(x)
        logger.debug(f"\t\tx = self.conv1(x)")
        logger.debug(f"\t\t// self.conv1: {self.conv1}")
        logger.debug(f"\t\t// x.shape: {x.shape}\n")

        ## for debug
        file_path = f"./npy_save/backbone_body_stem_conv1_output.npy"
        arr = x.cpu().numpy()
        np.save(file_path, arr)
        logger.debug(f"\t\tstem conv1 output of shape {arr.shape} saved into {file_path}.npy\n\n")

        x = self.bn1(x)
        logger.debug(f"\t\tx = self.bn1(x)")
        logger.debug(f"\t\t// self.bn1: {self.bn1}")
        logger.debug(f"\t\t// x.shape: {x.shape}\n")

        ## for debug
        file_path = f"./npy_save/backbone_body_stem_bn1_output.npy"
        arr = x.cpu().numpy()
        np.save(file_path, arr)
        logger.debug(f"\t\tstem bn1 output of shape {arr.shape} saved into {file_path}.npy\n\n")

        x = F.relu_(x)
        logger.debug(f"\t\tx = F.relu_(x)")
        logger.debug(f"\t\t// x.shape: {x.shape}\n")

        ## for debug
        file_path = f"./npy_save/backbone_body_stem_relu_output.npy"
        arr = x.cpu().numpy()
        np.save(file_path, arr)
        logger.debug(f"\t\tstem relu output of shape {arr.shape} saved into {file_path}.npy\n\n")
        """
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        logger.debug(f"\t\tx = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)")
        logger.debug(f"\t\t// x.shape: {x.shape}\n")
        """
        # https://github.com/xxradon/PytorchToCaffe/issues/16
        # to mimic caffe pooling's round_mode (ceiling)
        #x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        """
        https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        When ceil_mode=True, sliding windows are allowed to go off-bounds 
        if they start within the left padding or the input. 
        Sliding windows that would start in the right padded region are ignored.
        """
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        logger.debug(f"\t\tx = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)")
        logger.debug(f"\t\t// x.shape: {x.shape}\n")

        ## for debug
        file_path = f"./npy_save/backbone_body_stem_maxpool_output.npy"
        arr = x.cpu().numpy()
        np.save(file_path, arr)
        logger.debug(f"\t\tstem maxpool output of shape {arr.shape} saved into {file_path}.npy\n\n")

        logger.debug(f"\treturn x\n")
        logger.debug(f"\n\t}} // END BaseStem.forward()")
        return x


# ------------------------------------------
# StemWithFixedBatchNorm- inherits BaseStem
# ------------------------------------------
class StemWithFixedBatchNorm(BaseStem):


    def __init__(self, cfg):

        logger.debug(f"\n\tStemWithFixedBatchNorm.__init__() {{ //BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")

        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )
        logger.debug(f"\n\t}} // END StemWithFixedBatchNorm.__init__()")


# ------------------------------------------------
# Register Modules defined above
# ------------------------------------------------
logger.debug(f"# ======================")
logger.debug(f"# Registered Modules")
logger.debug(f"# ======================")
logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}")

_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
})

logger.debug(f"\t// _TRANSFORMATION_MODULES: {_TRANSFORMATION_MODULES}\n" )

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
})
logger.debug(f"\t// _STEM_MODULES : {_TRANSFORMATION_MODULES}\n" )

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})

logger.debug(f"\n# Networks Stage Specs:")
for nw_name, specs in _STAGE_SPECS.items():
    logger.debug(f"\t'{nw_name}':")
    logger.debug(f"\t(")
    for  spec in specs:
        logger.debug(f"\t\t{spec}")
    logger.debug(f"\t)\n")

logger.debug(f"\n\n")
