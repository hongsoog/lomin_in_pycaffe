# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

# for model debugging log
import logging
from model_log import  logger

class FPN(nn.Module):
    """
    Add a series of feature maps (actually the output of the last layer of stage 2~5 in ResNet) into FPN
    The depth of these feature maps is assumed to be continuously increasing,
    and the feature maps must be continuous (from a stage perspective)
    """

    def __init__( self, in_channels_list, out_channels, conv_block, top_blocks=None ):
        """
        Arguments:
            in_channels_list (list[int]):
                indicates the number of channels of each feature map that will be fed to FPN
            out_channels (int):
                number of channels of the FPN representation
                all feature maps will eventually be converted to this number of channels
            conv_block
                conv_with_kaiming_uniform(cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU),
            top_blocks (nn.Module or None):
               top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
                if top_blocks is provided, it will be at the end of FPN
                an extra op is performed on the last (smallest resolution) FPN output
                and then the result will be expanded into a result list to return
        """
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n=========================================== FPN.__init__ begin")
            logger.debug(f"\t======constructor params")
            logger.debug(f"\t\tin_channels_list: {in_channels_list}")
            logger.debug(f"\t\tout_channels: {out_channels}")
            # conv_block: function conv_with_kaiming_unifor.make_cov
            # defined in maskrcnn_benchmark/modelling/make_layers.py
            logger.debug(f"\t\tconv_block: {conv_block}")
            logger.debug(f"\t\ttop_blocks: {top_blocks}")
            logger.debug(f"\t======constructor params")

            logger.debug("\tsuper(FPN, self).__init__()\n")

        super(FPN, self).__init__()

        # create two empty lists
        self.inner_blocks = []
        self.layer_blocks = []

        # Assuming we are using ResNet-50-FPN and configuration,
        # the value of in_channels_list is: [0, 512, 1024, 2048]
        logger.debug(f"\tfor idx, in_channels in enumerate(in_channels_list, 1):")
        for idx, in_channels in enumerate(in_channels_list, 1):

            # note that index starts from 1
            if logger.level == logging.DEBUG:
                logger.debug(f"\n\t\t==> iteration with idx:{idx}, in_channels:{in_channels}")

            # naming rule for inner block : "fpn_inner" with index as suffix
            # ex. fpn_inner1, fpn_inner2, fpn_inner3, fpn_inner4
            inner_block = "fpn_inner{}".format(idx)

            # naming rule for layer block : "fpn_layer" with index as suffix
            # ex. fpn_layer1, fpn_layer2, fpn_layer3, fpn_layer4
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                if logger.level == logging.DEBUG:
                    logger.debug(f"\t\tif in_channels =={in_channels}, skip\n")
                continue

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tinner_block: {inner_block}")
                logger.debug(f"\t\tlayer_block: {layer_block}")
            # create inner_block_module,  where
            #    in_channels is the number of channels output by each stage
            #    out_channels is 256, defined in the user configuration file.
            #
            # The size of the convolution kernel here is 1, and the main function
            # of the convolution layer is to change the number of channels to out_channels
            # for dimension reduction
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tinner_block_module = conv_block(in_channels={in_channels}, out_channels={out_channels}, 1)")
            inner_block_module = conv_block(in_channels, out_channels, 1)

            # after changing the channels, perform a 3×3 convolution calculation
            # on the feature map of each stage, and the number of channels remains unchanged
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tlayer_block_module = conv_block(out_channels={out_channels}, out_channels={out_channels}, 3,1)")
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)

            # Add the current feature map into FPN
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tself.add_module({inner_block}, inner_block_module)")

            self.add_module(inner_block, inner_block_module)

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tself.add_module({layer_block}, layer_block_module)")

            self.add_module(layer_block, layer_block_module)

            # Add the name of the FPN module of the current stage
            # to the corresponding list (inner_blocks and layer_blocks respectively)
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tself.inner_blocks.append({inner_block})")

            self.inner_blocks.append(inner_block)

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tself.layer_blocks.append({layer_block})")

            self.layer_blocks.append(layer_block)

        # Use top_blocks as a member variable of the FPN class
        self.top_blocks = top_blocks

        # debug
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\tself.inner_blocks: {self.inner_blocks}")
            logger.debug(f"\t\tself.fpn_inner2: {self.fpn_inner2}")
            logger.debug(f"\t\tself.fpn_inner3: {self.fpn_inner3}")
            logger.debug(f"\t\tself.fpn_inner4: {self.fpn_inner4}")
            logger.debug(f"\n\tself.layer_blocks: {self.layer_blocks}")
            logger.debug(f"\t\tself.fpn_layer2: {self.fpn_layer2}")
            logger.debug(f"\t\tself.fpn_layer3: {self.fpn_layer3}")
            logger.debug(f"\t\tself.fpn_layer4: {self.fpn_layer4}")
            logger.debug(f"\n\tself.top_blocks: {self.top_blocks}")
            logger.debug(f"=========================================== FPN.__init__ end\n\n")

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]):
                feature maps of each feature level,
                The calculation result of ResNet just meets the input requirements of FPN,
                so you can use nn.Sequential to directly combine the two

        Returns:
             results (tuple[Tensor]):
                 tuple of list of feature maps after FPN, the order is high-resolution first
        """

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\nFPN.forward(self,x) ====== BEGIN")
            logger.debug(f"\t======forward param: x  = [C2, C3, C4, C5] ")
        # first, calculate the FPN result of the last layer (lowest resolution) feature map.

        if logger.level == logging.DEBUG:
            logger.debug(f"\tlen(x) = {len(x)}")
            for idx, element in enumerate(x):
                logger.debug(f"\tC[{idx+1}].shape : {element.shape}")
            logger.debug(f"\n\tx[-1].shape = {x[-1].shape}")

        # last_inner = fpn_inner4(C4)
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\tlast_inner = {self.inner_blocks[-1]}(C4)")
            logger.debug(f"\t\tself.innerblocks[-1] = {getattr(self, self.inner_blocks[-1])}")
            logger.debug(f"\t\tlast_inner.shape = {last_inner.shape}\n")

        # create an empty result list
        results = []
        """
        logger.debug(f"\tresults = []")
        """

        # add the calculation result of the last layer to results
        # i.e
        # results.append( fpn_layer4(last_inner)
        # == results.append( fpn_layer4(fpn_inner4(C4))
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\tresults.append({self.layer_blocks[-1]}(last_inner))")
            logger.debug(f"\t\tself.layer_blocks[-1]: {getattr(self, self.layer_blocks[-1])}")
            logger.debug(f"\t\tresults[0].shape: {results[0].shape}\n")

        # [:-1] get the first three items,
        # [::-1] represents slice from beginning to end, the step size is -1, the effect is the list inversion
        # For example, the operation result of self.inner_block[:-1][::-1] in zip is
        # [fpn_inner3, fpn_inner2, fpn_inner1], which is equivalent to inverting the list
        logger.debug("\tfor feature, inner_block, layer_block" +
              "\n\t\t\tin zip[(x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):\n")

        it_num = 0

        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t====================================")
                logger.debug(f"\t\titeration {it_num} summary")
                logger.debug(f"\t\t====================================")
                logger.debug(f"\t\tfeature.shape: {feature.shape}")
                logger.debug(f"\t\tinner_block: {inner_block} ==> {getattr(self, inner_block)}")
                logger.debug(f"\t\tlayer_block: {layer_block} ==> {getattr(self, layer_block)}")
                logger.debug(f"\t\tlast_inner.shape: {last_inner.shape}")
                logger.debug(f"\t\t====================================\n")

            if not inner_block:
                logger.debug(f"\t\tif not inner_block: continue \n")
                continue

            # the feature map is zoomed in/out according to the given scale parameter,
            # where scale=2, so it is zoomed in
            eltwise_suffix = inner_block[-1]

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t--------------------------------------------------")
                logger.debug(f"\t\t{it_num}.1 Upsample : replace with Decovolution in caffe")
                logger.debug(f"\t\tlayer name in caffe: {inner_block}_upsample = Deconvolution(last_inner)")
                logger.debug(f"\t\t--------------------------------------------------")
                logger.debug(f"\t\tinner_top_down = interpolate(last_inner, scale_factor=2, mode='nearest')\n")
            # Updample
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tlast_inner.shape: {last_inner.shape}")
                logger.debug(f"\t\tinner_top_down.shape : {inner_top_down.shape}")
                logger.debug(f"\t\t--------------------------------------------------\n")

            # get the calculation result of inner_block
            inner_lateral = getattr(self, inner_block)(feature)

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t--------------------------------------------------")
                logger.debug(f"\t\t{it_num}.2 inner_lateral = {getattr(self, inner_block)}(feature)")
                logger.debug(f"\t\tlayer name in caffe: {inner_block}_lateral={getattr(self, inner_block)}(feature)")
                logger.debug(f"\t\t--------------------------------------------------")
                logger.debug(f"\t\t\tinner_block: {inner_block} ==> {getattr(self, inner_block)}")
                logger.debug(f"\t\t\tinput: feature.shape: {feature.shape}")
                logger.debug(f"\t\t\toutput: inner_lateral.shape: {inner_lateral.shape}\n")
                logger.debug(f"\t\t--------------------------------------------------\n")

            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)

            # Superimpose the two as the output of the current pyramid level and
            # use it as an input to the next pyramid level
            last_inner = inner_lateral + inner_top_down

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t--------------------------------------------------")
                logger.debug(f"\t\t{it_num}.3 Elementwise Addition: replaced with eltwise in caffe")
                logger.debug(f"\t\tlayer in caffe: eltwise_{eltwise_suffix} = eltwise({inner_block}_lateral, {inner_block}_upsample )")
                logger.debug(f"\t\t--------------------------------------------------")
                logger.debug(f"\t\tlast_inner = inner_lateral + inner_top_down")
                logger.debug(f"\t\t\tinner_lateral.shape: {inner_lateral.shape}")
                logger.debug(f"\t\t\tinner_top_down.shape: {inner_top_down.shape}")
                logger.debug(f"\t\t\tlast_inner.shape : {last_inner.shape}")
                logger.debug(f"\t\t--------------------------------------------------\n")

            # Add the current pyramid level output to the result list,
            # Note that use layer_block to perform convolution calculations at the same time,
            # in order to make the highest resolution first, we need to insert the current
            # pyramid level output to the 0 position (i.e, prepend)
            results.insert(0, getattr(self, layer_block)(last_inner))

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t--------------------------------------------------")
                logger.debug(f"\t\t{it_num}.4 results.insert(0, {getattr(self,layer_block)}(last_inner)")
                logger.debug(f"\t\tlayer in caffe: {layer_block} = {getattr(self, layer_block)}(eltwise_{eltwise_suffix})")
                logger.debug(f"\t\t--------------------------------------------------")
                logger.debug(f"\t\t\tlayer_block: {layer_block} ==> {getattr(self, layer_block)}")
                logger.debug(f"\t\t\tinput: last_inner.shape = {last_inner.shape}")
                logger.debug(f"\t\t--------------------------------------------------\n")

                logger.debug(f"\t\t--------------------------------------------------")
                logger.debug(f"\t\tresults after iteration {it_num}")
                logger.debug(f"\t\t--------------------------------------------------")
                for idx, r in enumerate(results):
                    logger.debug(f"\t\t\tresults[{idx}].shape: {r.shape}")

                logger.debug(f"\t\t--------------------------------------------------\n")
                it_num += 1

        if logger.level == logging.DEBUG:
            logger.debug(f"\tfor loop END\n")

        # if top_blocks is not empty, execute these additional ops
        if isinstance(self.top_blocks, LastLevelP6P7):
            # LastLevelP6P7: generate extra layers, P6 and P7 in Retinanet
            if logger.level == logging.DEBUG:
                logger.debug(f"\n\tif isinstance(self.top_blocks, LastLevelP6P7):")
                logger.debug(f"\t\tlast_result = self.top_blocks(x[-1], results[-1])")
            last_results = self.top_blocks(x[-1], results[-1])

            # append the newly calculated result to the list
            results.extend(last_results)

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tresults.extend(last_results)")


        elif isinstance(self.top_blocks, LastLevelMaxPool):
            if logger.level == logging.DEBUG:
                logger.debug(f"\tif isinstance(self.top_blocks, LastLevelMaxpool):")

            last_results = self.top_blocks(results[-1])

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tlast_results = self.top_blocks(results[-1])")

            results.extend(last_results)
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tresults.extend(last_results)")

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\t\tresults")
            for idx, r in enumerate(results):
                logger.debug(f"\t\tresult[{idx}].shape: {r.shape}")
            logger.debug(f"\n\treturn tuple(results)")

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\nFPN.forward(self,x) ====== END")
        # return as a tuple (read-only)
        return tuple(results)

# The last level of max pool layer
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        if logger.level == logging.DEBUG:
            logger.debug(f"LastLevelMaxPool.forward")
            logger.debug(f"return [F.max_pool2d(x, 1, 2, 0)]")

        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\n\t\tLastLevelP6P7.__init__(self, in_channels={in_channels}, out_channels={out_channels}) ====== BEGIN")
            logger.debug(f"\t\t\tsuper(LastLevelP6P7, self).__init__()")

        super(LastLevelP6P7, self).__init__()

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\tself.p6 = nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, 3, 2, 1)")
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\tself.p7 = nn.Conv2d(out_channels={out_channels}, out_channels={out_channels}, 3, 2, 1)")
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\tfor module in [self.p6, self.p7]:")

        for module in [self.p6, self.p7]:
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t\t\tmodule={module}")
                logger.debug(f"\t\t\t\tnn.init.kaiming_uniform_(module.weight=module.weight, a=1)")
            nn.init.kaiming_uniform_(module.weight, a=1)

            if logger.level == logging.DEBUG:
                logger.debug("\t\t\t\tnn.init.constant_(module.bias=module.bias, 0)")
            nn.init.constant_(module.bias, 0)

        self.use_P5 = in_channels == out_channels
        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\tself.use_p5 : {self.use_P5}")
            logger.debug(f"\n\t\tLastLevelP6P7.__init__(self, in_channels={in_channels}, out_channels={out_channels}) ====== END\n\n")


    def forward(self, c5, p5):
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\t\tLastLevelP6P7.forward(self, c5, p5) ============= BEGIN ")
            logger.debug(f"\t\t\tc5.shape: {c5.shape}")
            logger.debug(f"\t\t\tp5.shape: {p5.shape}\n")

        # for debug
        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\tif (self.use_P5 == {self.use_P5})")

            if (self.use_P5):
                logger.debug("\t\t\t\tx=p5")
            else:
                logger.debug("\t\t\t\tx=c5")

        x = p5 if self.use_P5 else c5

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\tx.shape = {x.shape}")

        p6 = self.p6(x)

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\tp6 = self.p6(x)")
            logger.debug(f"\t\t\t\tself.p6: {self.p6}")
            logger.debug(f"\t\t\t\tp6.shape: {p6.shape}\n")

        p7 = self.p7(F.relu(p6))
        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\tp7 = self.p7(F.relu(p6))")
            logger.debug(f"\t\t\t\tself.p7: {self.p7}")
            logger.debug(f"\t\t\t\tp7.shape: {p7.shape}\n")
            logger.debug(f"\t\t\treturns [p6, p7]")
            logger.debug(f"\t\tLastLevelP6P7.forward(self, c5, p5) ============= END\n\n")
        return [p6, p7]
