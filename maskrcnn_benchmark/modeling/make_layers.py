# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import inspect
import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.modeling.poolers import Pooler

# for model debugging log
from model_log import  logger

def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn




def make_conv3x3(
    in_channels,
    out_channels,
    dilation=1,
    stride=1,
    use_gn=False,
    use_relu=False,
    kaiming_init=True,
):
    # TODO: config check
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False if use_gn else True,
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if use_gn:
        module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


def make_fc(dim_in, hidden_dim, use_gn=False):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''
    logger.debug(f"make_fc(dim_in={dim_in}, hidden_dim={hidden_dim}, use_gn={use_gn}) called")
    if use_gn:
        logger.debug(f"\tif use_gn = {use_gn}:")

        logger.debug(f"\t\tfc = nn.Linear(dim_in, hidden_dim, bias=False)")
        fc = nn.Linear(dim_in, hidden_dim, bias=False)

        logger.debug(f"\t\tnn.init.kaiming_uniform_(fc.weight, a=1)")
        nn.init.kaiming_uniform_(fc.weight, a=1)

        logger.debug(f"\t\treturn nn.Sequential(fc, group_norm(hidden_dim))")
        return nn.Sequential(fc, group_norm(hidden_dim))

    logger.debug(f"\tuse_gn = {use_gn}")
    logger.debug(f"\tfc = nn.Linear(dim_in, hidden_dim)")
    fc = nn.Linear(dim_in, hidden_dim)
    logger.debug(f"\tnn.init.kaiming_uniform_(fc.weight, a=1)")
    nn.init.kaiming_uniform_(fc.weight, a=1)
    logger.debug(f"\tnn.init.constant_(fc.bias, 0)")
    nn.init.constant_(fc.bias, 0)

    logger.debug(f"\treturn fc")
    return fc

def conv_with_kaiming_uniform(use_gn=False, use_relu=False):

    logger.debug(f"\n\t\t\tconv_with_kaiming_uniform(use_gn={use_gn}, use_relut={use_relu}) {{ //BEGIN")
    logger.debug(f"\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")

    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        logger.debug(f"\n\t\t\t\tmake_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1) {{ //BEGIN")
        logger.debug(f"\t\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")
        # TODO: config check
        logger.debug(f"\t\t\t\tParams:")
        logger.debug(f"\t\t\t\t\tin_channels: {in_channels}")
        logger.debug(f"\t\t\t\t\tout_channels: {out_channels}")
        logger.debug(f"\t\t\t\t\tkernel_size: {kernel_size}")
        logger.debug(f"\t\t\t\t\tstride: {stride}")
        logger.debug(f"\t\t\t\t\tdilation: {dilation}")

        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True,
        )

        """
        logger.debug(f"\t\t\t\tuse_gn: {use_gn}")
        logger.debug(f"\t\t\t\tuse_relu: {use_relu}")
        logger.debug(f"\t\t\t\tin_channels: {in_channels}")
        logger.debug(f"\t\t\t\tout_channels: {out_channels}")
        logger.debug(f"\t\t\t\tkernel_size: {kernel_size}")
        logger.debug(f"\t\t\t\tstride: {stride}")
        logger.debug(f"\t\t\t\tdilation: {dilation}")
        """

        logger.debug(f"\t\t\t\tconv = Conv2d(in_channles={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, stride={stride}")
        logger.debug(f"\t\t\t\t       padding={dilation*(kernel_size -1) //2}, dilation={dilation}, bias={False if use_gn else True}, )")
        logger.debug(f"\t\t\t\tconv: {conv}")


        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        logger.debug(f"\t\t\t\tnn.init.kaiming_uniform_(conv.weight, a=1)")

        if not use_gn:
            logger.debug(f"\t\t\t\tif not use_gn:")
            nn.init.constant_(conv.bias, 0)
            logger.debug(f"\t\t\t\t\tnn.init.constant_(conv.bias, 0)")

        module = [conv,]
        logger.debug(f"\t\t\t\tmodule = [conv,]")
        logger.debug(f"\t\t\t\tmodule: {module}")

        if use_gn:
            logger.debug(f"\t\t\t\tif use_gn:")
            logger.debug(f"\t\t\t\t\tmodule.append(group_norm(out_channels={out_channels}))")
            module.append(group_norm(out_channels))

        if use_relu:
            logger.debug(f"\t\t\t\tif use_relu:")
            logger.debug(f"\t\t\t\t\tmodule.append(nn.ReLU(inplace=True))")
            module.append(nn.ReLU(inplace=True))

        if len(module) > 1:
            logger.debug(f"\t\t\t\tif len(module) > 1:")
            logger.debug(f"\t\t\t\t\tmodule: {module}")
            logger.debug(f"\t\t\t\t\treturn nn.Sequential(*module)")
            return nn.Sequential(*module)

        logger.debug(f"\t\t\t\tconv: {conv}")
        logger.debug(f"\t\t\t\treturn conv\n")
        logger.debug(f"\t\t\t\t}} // END conv_with_kaiming_uniform().make_conv()\n")
        return conv

    logger.debug(f"\t\t\t}} // END conv_with_kaiming_uniform(use_gn={use_gn}, use_relu={use_relu})\n")
    return make_conv
