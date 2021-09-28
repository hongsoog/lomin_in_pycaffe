# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import inspect
import logging
from collections import OrderedDict
from torch import nn
import timm

# registry is used to manage the registration of the module,
# so that the module can be used like a dictionary
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import resnet
from . import fpn as fpn_module

# for model debugging log
from model_log import  logger

TIMM_MODEL_MAP = {
    "TIMM-MOBILENETV2-100": "mobilenetv2_100",
}
def build_timm_backbone(cfg):
    model_name = TIMM_MODEL_MAP[cfg.MODEL.BACKBONE.CONV_BODY]
    body = timm.create_model(
        model_name=model_name, 
        pretrained=cfg.MODEL.TIMM.USE_PRETRAINED,
        features_only=True,
        out_indices=cfg.MODEL.TIMM.OUT_INDICES
    )
    if cfg.MODEL.RPN.USE_FPN:
        # FIXME:
        if cfg.MODEL.RECOGNITION:
            in_channels = body.feature_channels()
        else:
            in_channels = [0] + body.feature_channels()
        out_channels = cfg.MODEL.TIMM.BACKBONE_OUT_CHANNELS
        in_channels_p6p7 = in_channels[-1] if cfg.MODEL.RETINANET.USE_C5 else out_channels
        fpn = fpn_module.FPN(
            in_channels_list=in_channels,
            out_channels=out_channels,
            conv_block=conv_with_kaiming_uniform(
                cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
            ),
            top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
        )
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
        model.out_channels = out_channels
    else:
        # body = timm.create_model(
        #     model_name=model_name, 
        #     pretrained=cfg.MODEL.TIMM.USE_PRETRAINED,
        # )
        model = nn.Sequential(OrderedDict([("body", body)]))
        model.out_channels = cfg.MODEL.TIMM.BACKBONE_OUT_CHANNELS
    return model

for k in TIMM_MODEL_MAP.keys():
    registry.BACKBONES.register(k, build_timm_backbone)

# Create resent + fpn (feature pyramid network) network, according to the configuration information,
# it will be called by build_backbone() function
# from maskrcnn_benchmark.modeling import registry
@registry.BACKBONES.register("R-50-FPN-RETINANET")  # ResMet50, FPN, RetinaNet as RPN
@registry.BACKBONES.register("R-101-FPN-RETINANET") # ResNet101, FPN, RetinaNet as RPN
def build_resnet_fpn_p3p7_backbone(cfg):

    logger.debug(f"\n\tbuild_resnet_fpn_p3p7_backbone(cfg) {{ // BEGIN")
    logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
    logger.debug(f"\t\tParams:")
    logger.debug(f"\t\t\tcfg:")

    # -------------------------------------
    #  1. backbone.body  : ResNet
    # -------------------------------------
    # create resnet network first
    # class ResNet(cfg) defined in the resnet.py file

    logger.debug(f"\t\tbody = resnet.ResNet(cfg) // CALL")

    body = resnet.ResNet(cfg)

    logger.debug(f"\t\tbody = resnet.ResNet(cfg) // RETURNED")


    # get the channels parameters required by fpn
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS

    logger.debug(f"\t\tcfg.MODEL.RESNETS.RES2_OUT_CHANNELS: {cfg.MODEL.RESNETS.RES2_OUT_CHANNELS}")
    logger.debug(f"\t\tin_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS")
    logger.debug(f"\t\tin_channels_stage2: {in_channels_stage2}")

    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS

    logger.debug(f"\t\tcfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS:{cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS}")
    logger.debug(f"\t\tout_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS")
    logger.debug(f"\t\tout_channels: {out_channels}")

    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 else out_channels

    logger.debug(f"\t\tin_channels_stage2: {in_channels_stage2 }")
    logger.debug(f"\t\tout_channels: {out_channels}")
    logger.debug(f"\t\tcfg.MODEL.RETINANET.USE_C5: {cfg.MODEL.RETINANET.USE_C5:}")
    logger.debug(f"\t\tin_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 else out_channels")
    logger.debug(f"\t\tin_channels_p6p7: {in_channels_p6p7}")


    # -------------------------------------
    #  2. backbone.fpn
    # -------------------------------------
    # create an fpn network
    # class FPN defined in fpn.py

    logger.debug("\n\t\tfpn = fpn_module.FPN(" )
    logger.debug(f"\n\t\t\t\tin_channels_list = [0, {in_channels_stage2*2}, {in_channels_stage2*4}, {in_channels_stage2 * 8}],")
    logger.debug(f"\n\t\t\t\tout_channels = {out_channels}, ")
    logger.debug(f"\n\t\t\t\tconv_block=conv_with_kaiming_uniform( cfg.MODEL.FPN.USE_GN ={cfg.MODEL.FPN.USE_GN }, cfg.MODEL.FPN.USE_RELU ={cfg.MODEL.FPN.USE_RELU} ),")
    logger.debug(f"\n\t\t\t\ttop_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7={in_channels_p6p7}, out_channels={out_channels},) // CALL")

    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )

    logger.debug("\n\t\tfpn = fpn_module.FPN(" )
    logger.debug(f"\n\t\t\t\tin_channels_list = [0, {in_channels_stage2*2}, {in_channels_stage2*4}, {in_channels_stage2 * 8}],")
    logger.debug(f"\n\t\t\t\tout_channels = {out_channels}, ")
    logger.debug(f"\n\t\t\t\tconv_block=conv_with_kaiming_uniform( cfg.MODEL.FPN.USE_GN ={cfg.MODEL.FPN.USE_GN }, cfg.MODEL.FPN.USE_RELU ={cfg.MODEL.FPN.USE_RELU} ),")
    logger.debug(f"\n\t\t\t\ttop_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7={in_channels_p6p7}, out_channels={out_channels},) // RETURNED")
    logger.debug("f\t\t\tfpn: {fpn}")

    logger.debug(f'\t\tmodel = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) // CALL')

    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))

    logger.debug(f'\t\tmodel = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) // RETURNED')
    logger.debug(f"\t\tmodel: {model}")

    model.out_channels = out_channels
    logger.debug(f"\t\tmodel.out_channels = out_channels")
    logger.debug(f"\t\t\tmodel.out_channels: {model.out_channels:}")

    logger.debug(f"\t\treturn model")
    logger.debug(f"\t}} // END build_resnet_fpn_p3p7_backbone(cfg) \n\n")
    return model

def build_backbone(cfg):
    if logger.level == logging.DEBUG:
        logger.debug(f"\n\tbuild_backbone(cfg) {{ // BEGIN")
        logger.debug(f"\t\tdefined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\tParams:")
        logger.debug(f"\t\t\tcfg:")
        #logger.debug(f"\tregistry.BACKBONES: {registry.BACKBONES}")
        #logger.debug(f"\tcfg.MODEL.BACKBONE.CONV_BODY: {cfg.MODEL.BACKBONE.CONV_BODY}")
        #logger.debug(f"\tcfg: {cfg}")
        logger.debug(f"\tregistry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg): {registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)}")
        logger.debug(f"\t\treturn registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)")
        logger.debug(f"\t}} // END build_backbone(cfg)\n")

    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
