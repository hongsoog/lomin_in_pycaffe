# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import inspect
import torch
import torch.nn.functional as F
from torch import nn

# for model debugging log
import logging
import  numpy as np
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
        logger.debug(f"\t\t# =================================")
        logger.debug(f"\t\t# 1-1-2-2 FPN build")
        logger.debug(f"\t\t# =================================\n")
        logger.debug(f"\n\nFPN.__init__ {{ // BEGIN")
        logger.debug(f"\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t// Params")
        logger.debug(f"\t\t\t// in_channels_list: {in_channels_list}")
        logger.debug(f"\t\t\t// out_channels: {out_channels}")
        # conv_block: function conv_with_kaiming_unifor.make_cov
        # defined in maskrcnn_benchmark/modelling/make_layers.py
        logger.debug(f"\t\t\t// conv_block: {conv_block}")
        logger.debug(f"\t\t\t// top_blocks: {top_blocks}\n")


        super(FPN, self).__init__()
        logger.debug("\t\tsuper(FPN, self).__init__()\n")

        # create two empty lists
        logger.debug("\t\t# create two empty lists")
        self.inner_blocks = []
        logger.debug("\t\tself.inner_blocks = []")

        self.layer_blocks = []
        logger.debug("\t\tself.layer_block = []")

        # Assuming we are using ResNet-50-FPN and configuration,
        # the value of in_channels_list is: [0, 512, 1024, 2048]
        logger.debug(f"\t\tfor idx, in_channels in enumerate(in_channels_list, 1) {{\n")
        total_iterations = len(in_channels_list)

        for idx, in_channels in enumerate(in_channels_list, 1):

            # note that index starts from 1
            logger.debug(f"\t\t\t{{")
            logger.debug(f"\t\t\t\t# -----------------------------------------------------")
            logger.debug(f"\t\t\t\t# in_channels:{in_channels}, iteration {idx}/{total_iterations} BEGIN")
            logger.debug(f"\t\t\t\t# -----------------------------------------------------")

            # naming rule for inner block : "fpn_inner" with index as suffix
            # ex. fpn_inner1, fpn_inner2, fpn_inner3, fpn_inner4
            inner_block = "fpn_inner{}".format(idx)
            logger.debug('\t\t\t\tinner_block = "fpn_inner{}".format(idx)')
            logger.debug("\t\t\t\t// inner_block: {inner_block}\n")

            # naming rule for layer block : "fpn_layer" with index as suffix
            # ex. fpn_layer1, fpn_layer2, fpn_layer3, fpn_layer4
            layer_block = "fpn_layer{}".format(idx)
            logger.debug('\t\t\t\tlayer_block = "fpn_layer{}".format(idx)')
            logger.debug('\t\t\t\t// layer_block: {layer_block}\n')

            if in_channels == 0:
                logger.debug(f"\t\t\t\tif in_channels =={in_channels}, skip\n")
                logger.debug(f"\t\t\t}}\n\t\t\t# iteration {idx}/{total_iterations} END\n")
                continue

            logger.debug(f"\t\t\t\t// inner_block: {inner_block}")
            logger.debug(f"\t\t\t\t// layer_block: {layer_block}\n")
            # create inner_block_module,  where
            #    in_channels is the number of channels output by each stage
            #    out_channels is 256, defined in the user configuration file.
            #
            # The size of the convolution kernel here is 1, and the main function
            # of the convolution layer is to change the number of channels to out_channels
            # for dimension reduction
            inner_block_module = conv_block(in_channels, out_channels, 1)
            logger.debug(f"\t\t\t\tinner_block_module = conv_block(in_channels={in_channels}, out_channels={out_channels}, 1)")
            logger.debug(f"\t\t\t\t// inner_block_module: {inner_block_module}\n")

            # after changing the channels, perform a 3×3 convolution calculation
            # on the feature map of each stage, and the number of channels remains unchanged
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            logger.debug(f"\t\t\t\tlayer_block_module = conv_block(out_channels={out_channels}, out_channels={out_channels}, 3,1)")
            logger.debug(f"\t\t\t\t// layer_block_module: {layer_block_module}\n")

            # Add the current feature map into FPN

            self.add_module(inner_block, inner_block_module)
            logger.debug(f"\t\t\t\tself.add_module(inner_block, inner_block_module)\n")


            self.add_module(layer_block, layer_block_module)
            logger.debug(f"\t\t\t\tself.add_module(layer_block, layer_block_module)\n")

            # Add the name of the FPN module of the current stage
            # to the corresponding list (inner_blocks and layer_blocks respectively)

            self.inner_blocks.append(inner_block)
            logger.debug(f"\t\t\t\tself.inner_blocks.append({inner_block})\n")

            self.layer_blocks.append(layer_block)
            logger.debug(f"\t\t\t\tself.layer_blocks.append({layer_block})\n")
            logger.debug(f"\t\t\t}}\n\t\t\t# iteration {idx}/{total_iterations} END\n")

        logger.debug(f"\n\t\t}} // END for idx, in_channels in enumerate(in_channels_list, 1)")


        # Use top_blocks as a member variable of the FPN class
        self.top_blocks = top_blocks
        logger.debug(f"\t\tself.top_blocks = top_blocks\n")

        # debug
        logger.debug(f"\t\t// self.inner_blocks: {self.inner_blocks}")
        logger.debug(f"\t\t// self.fpn_inner2: {self.fpn_inner2}")
        logger.debug(f"\t\t// self.fpn_inner3: {self.fpn_inner3}")
        logger.debug(f"\t\t// self.fpn_inner4: {self.fpn_inner4}")
        logger.debug(f"\n\t// self.layer_blocks: {self.layer_blocks}")
        logger.debug(f"\t\t// self.fpn_layer2: {self.fpn_layer2}")
        logger.debug(f"\t\t// self.fpn_layer3: {self.fpn_layer3}")
        logger.debug(f"\t\t// self.fpn_layer4: {self.fpn_layer4}")
        logger.debug(f"\n\t// self.top_blocks: {self.top_blocks}")
        logger.debug(f"\n}} // END FPN.__init__\n\n")

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

        logger.debug(f"\n\n")
        logger.debug(f"\t\t# =================================")
        logger.debug(f"\t\t# 2-3-1-2 FPN Forward")
        logger.debug(f"\t\t# =================================")

        logger.debug(f"\n\tFPN.forward(self,x) {{ // BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t// Param: x  = [C1, C2, C3, C4], return of Resnet.forward()")
        # first, calculate the FPN result of the last layer (lowest resolution) feature map.

        for idx, element in enumerate(x):
            logger.debug(f"\t\t\t// C[{idx+1}] of shape : {element.shape}")

        logger.debug(f"\n")
        logger.debug(f"\t\t\t# ===========================================================================")
        logger.debug(f"\t\t\t# FPN block info")
        logger.debug(f"\t\t\t# self.inner_blocks: {self.inner_blocks})")
        logger.debug(f"\t\t\t# self.layer_blocks: {self.layer_blocks})")
        logger.debug(f"\t\t\t# ===========================================================================\n")

        # last_inner = fpn_inner4(C4)
        logger.debug(f"\t\t\t# last_inner = fpn_inner4(C4)\n")

        logger.debug(f"\t\t\t# self.innerblocks[-1]:{self.inner_blocks[-1]}")
        logger.debug(f"\t\t\t# getattr(self, self.innerblocks[-1]):{getattr(self, self.inner_blocks[-1])}")
        logger.debug(f"\t\t\t// x[-1].shape = {x[-1].shape}\n")
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        logger.debug(f"\t\t\tlast_inner = getattr(self, self.inner_blocks[-1])(x[-1])")
        logger.debug(f"\t\t\t// last_inner.shape:{last_inner.shape}\n")

        file_path = f"./npy_save/{self.inner_blocks[-1]}_output"
        arr = last_inner.cpu().numpy()
        np.save(file_path, arr)
        logger.debug(f"\t\t\t# {self.inner_blocks[-1]} output of shape {arr.shape} saved into {file_path}.npy\n\n")

        # create an empty result list
        results = []
        logger.debug(f"\t\t\tresults = []")

        # add the calculation result of the last layer to results
        # i.e
        # results.append( fpn_layer4(last_inner)
        # == results.append( fpn_layer4(fpn_inner4(C4))
        logger.debug(f"\t\t\t# self.layer_blocks[-1]: {self.layer_blocks[-1]}")
        logger.debug(f"\t\t\t# getattr(self, self.layer_blocks[-1]): {getattr(self, self.layer_blocks[-1])}")
        logger.debug(f"\t\t\t// last_inner.shape: {last_inner.shape}]\n")

        results.append(getattr(self, self.layer_blocks[-1])(last_inner))

        logger.debug(f"\n\t\tresults.append(self.layer_blocks[-1](last_inner))")
        logger.debug(f"\t\t\t# results.append() : P4")
        logger.debug(f"\t\t// results[-1].shape: {results[-1].shape} <=== P4\n")

        file_path =f"./npy_save/{self.layer_blocks[-1]}_output"
        arr = results[0].cpu().numpy()
        np.save(file_path, arr)
        logger.debug(f"\t\t\t# {self.layer_blocks[-1]} output (P4) of shape {arr.shape} saved into {file_path}.npy\n\n")

        # [:-1] get the first three items,
        # [::-1] represents slice from beginning to end, the step size is -1, the effect is the list inversion
        # For example, the operation result of self.inner_block[:-1][::-1] in zip is
        # [fpn_inner3, fpn_inner2, fpn_inner1], which is equivalent to inverting the list
        logger.debug(f"\t\t\tfor feature, inner_block, layer_block in zip(")
        logger.debug(f"\t\t\t\t[(x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]): {{\n")

        it_num = 0

        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            logger.debug(f"\t\t\t\t{{ // BEGIN iteratrion for calc P{3-it_num}\n")

            logger.debug(f"\t\t\t\t# ====================================")
            logger.debug(f"\t\t\t\t# for calc P{3-it_num}")
            logger.debug(f"\t\t\t\t# feature.shape: {feature.shape}")
            logger.debug(f"\t\t\t\t# inner_block: {inner_block} ==> {getattr(self, inner_block)}")
            logger.debug(f"\t\t\t\t# layer_block: {layer_block} ==> {getattr(self, layer_block)}")
            logger.debug(f"\t\t\t\t# last_inner.shape: {last_inner.shape}")
            logger.debug(f"\t\t\t\t# ====================================\n")

            if not inner_block:
                logger.debug(f"\t\tif not inner_block: continue \n")
                continue

            # the feature map is zoomed in/out according to the given scale parameter,
            # where scale=2, so it is zoomed in

            # for generating caffe version layer name
            eltwise_suffix = inner_block[-1]
            logger.debug(f"\t\t\t\teltwise_suffix = inner_block[-1]")
            logger.debug(f"\t\t\t\t// eltwise_suffix: {eltwise_suffix}")

            #---------------
            # 1) Updample
            #---------------
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")

            logger.debug(f"\t\t\t\t# --------------------------------------------------")
            logger.debug(f"\t\t\t\t# for calc P{3-it_num}")
            logger.debug(f"\t\t\t\t# 1. Upsample : replace with Decovolution in caffe")
            logger.debug(f"\t\t\t\t# layer name in caffe: {inner_block}_upsample = Deconvolution(last_inner)")
            logger.debug(f"\t\t\t\t# --------------------------------------------------")
            logger.debug(f"\t\t\t\t// last_inner.shape: {last_inner.shape}")
            logger.debug(f"\t\t\t\tinner_top_down = F.interpolate(last_inner, scale_factor=2, mode='nearest')")
            logger.debug(f"\t\t\t\t// inner_top_down.shape : {inner_top_down.shape}\n")

            file_path = f"./npy_save/inner_top_down_for_{inner_block}"
            arr = inner_top_down.cpu().numpy()
            np.save(file_path, arr)
            logger.debug(f"\t\t\t\t# inner_top_down of shape {arr.shape} saved into {file_path}.npy\n\n")


            #---------------
            # 2) inner_block
            #---------------
            # get the calculation result of inner_block
            inner_lateral = getattr(self, inner_block)(feature)

            logger.debug(f"\t\t\t\t--------------------------------------------------")
            logger.debug(f"\t\t\t\t# for calc P{3-it_num}")
            logger.debug(f"\t\t\t\t# 2. inner_lateral = {getattr(self, inner_block)}(feature)")
            logger.debug(f"\t\t\t\t# layer name in caffe: {inner_block}_lateral={getattr(self, inner_block)}(feature)")
            logger.debug(f"\t\t\t\t# --------------------------------------------------")
            logger.debug(f"\t\t\t\t// inner_block: {inner_block} ==> {getattr(self, inner_block)}")
            logger.debug(f"\t\t\t\t// feature.shape: {feature.shape}")
            logger.debug(f"\t\t\t\tinner_lateral = getattr(self, inner_block)(feature)")
            logger.debug(f"\t\t\t\t// inner_lateral.shape: {inner_lateral.shape}\n")

            file_path = f"./npy_save/{inner_block}_output"
            arr = inner_lateral.cpu().numpy()
            np.save(file_path, arr)
            logger.debug(f"\t\t\t\t# {inner_block} output of shape {arr.shape} saved into {file_path}.npy\n\n")


            #----------------------------------------------------
            # 3) superimpose inner_block output with top_down
            #----------------------------------------------------
            # Superimpose the two as the output of the current pyramid level and
            # use it as an input to the next pyramid level
            last_inner = inner_lateral + inner_top_down

            logger.debug(f"\t\t\t\t# --------------------------------------------------")
            logger.debug(f"\t\t\t\t# for calc P{3-it_num}")
            logger.debug(f"\t\t\t\t# 3. Elementwise Addition: replaced with eltwise in caffe")
            logger.debug(f"\t\t\t\t# layer in caffe: eltwise_{eltwise_suffix} = eltwise({inner_block}_lateral, {inner_block}_upsample )")
            logger.debug(f"\t\t\t\t# --------------------------------------------------")
            logger.debug(f"\t\t\t\t// inner_lateral.shape: {inner_lateral.shape}")
            logger.debug(f"\t\t\t\t// inner_top_down.shape: {inner_top_down.shape}")
            logger.debug(f"\t\t\t\tlast_inner = inner_lateral + inner_top_down")
            logger.debug(f"\t\t\t\t// last_inner.shape : {last_inner.shape}")

            file_path = f"./npy_save/{inner_block}_ouptut_plus_inner_topdown"
            arr = last_inner.cpu().numpy()
            np.save(file_path, arr)
            logger.debug(f"\t\t\t\t# superimposing result of {inner_block} output plus inner topdown of shape {arr.shape} saved into {file_path}.npy\n\n")

            #----------------------------------------------------
            # 4) prepend result of layer_block on superimposed of inner_block output with top_down
            #----------------------------------------------------
            # Add the current pyramid level output to the result list,
            # Note that use layer_block to perform convolution calculations at the same time,
            # in order to make the highest resolution first, we need to insert the current
            # pyramid level output to the 0 position (i.e, prepend)

            results.insert(0, getattr(self, layer_block)(last_inner))

            logger.debug(f"\t\t\t\t# --------------------------------------------------")
            logger.debug(f"\t\t\t\t# for calc P{3-it_num}")
            logger.debug(f"\t\t\t\t# 4. results.insert(0, {getattr(self,layer_block)}(last_inner)")
            logger.debug(f"\t\t\t\t# layer in caffe: {layer_block} = {getattr(self, layer_block)}(eltwise_{eltwise_suffix})")
            logger.debug(f"\t\t\t\t# --------------------------------------------------")
            logger.debug(f"\t\t\t\t// layer_block: {layer_block} ==> {getattr(self, layer_block)}")
            logger.debug(f"\t\t\t\t// input: last_inner.shape = {last_inner.shape}")
            logger.debug(f"\t\t\t\tresults.insert(0, getattr(self, layer_block)(last_inner))")
            logger.debug(f"\t\t\t\t// results[0].shape: {results[0].shape}\n")

            file_path = f"./npy_save/{layer_block}_ouptut"
            arr = results[0].cpu().numpy()
            np.save(file_path, arr)
            logger.debug(f"\t\t\t\t# {layer_block} output (P{3 - it_num}) of shape {arr.shape} saved into {file_path}.npy\n\n")

            logger.debug(f"\t\t\t\t# --------------------------------------------------")
            logger.debug(f"\t\t\t\t# results after iteration {it_num}")
            logger.debug(f"\t\t\t\t# --------------------------------------------------")
            for idx, r in enumerate(results):
                logger.debug(f"\t\t\t\t# results[{idx}] of shape: {r.shape}")

            logger.debug(f"\t\t\t\t#--------------------------------------------------\n")
            logger.debug(f"\t\t\t\t}} // END iteratrion for calc P{3-it_num}\n")
            it_num += 1

        logger.debug(f"\t\t\t}} // for loop END\n")

        logger.debug(f"\t\t\t# --------------------------------------------------")
        logger.debug(f"\t\t\t# results after for loop")
        logger.debug(f"\t\t\t# --------------------------------------------------")
        for idx, r in enumerate(results):
            logger.debug(f"\t\t\t\t# results[{idx}] AKA P{idx + 2} of shape: {r.shape}")

        logger.debug(f"\n")

        # if top_blocks is not empty, execute these additional ops
        if isinstance(self.top_blocks, LastLevelP6P7):
            # LastLevelP6P7: generate extra layers, P6 and P7 in Retinanet

            logger.debug(f"\n\t\t\tif isinstance(self.top_blocks, LastLevelP6P7):")
            logger.debug(f"\t\t\t\t// self.top_blocks: {self.top_blocks}")
            logger.debug(f"\t\t\t\t// len(x): {len(x)}")
            for idx, element in enumerate(x):
                logger.debug(f"\t\t\t\t\t// x[{idx}].shape : {element.shape} ==> C{idx+1}")
            logger.debug(f"\t\t\t\t\t//x[-1] AKA C4 of shape : {x[-1].shape}\n\n")

            logger.debug(f"\t\t\t\t// len(results): {len(results)}")
            for idx, element in enumerate(results):
                logger.debug(f"\t\t\t\t\t// results[{idx}].shape : {element.shape} ==> P{idx+2}")
            logger.debug(f"\n\t\t\t\t// results[-1] AKA P4 of shape: {results[-1].shape}\n\n")

            logger.debug(f"\t\t\t\tlast_result = self.top_blocks(x[-1]==>C4, results[-1]==>P4) {{ // CALL")

            last_results = self.top_blocks(x[-1], results[-1])

            logger.debug(f"\t\t\t\t}}")
            logger.debug(f"\t\t\t\tlast_result = self.top_blocks(x[-1]==>C4, results[-1]==>P4) // RETURNED\n")
            logger.debug(f"\t\t\t\t// len(last_results):{len(last_results)}")
            for idx, element in enumerate(last_results):
                logger.debug(f"\t\t\t\t//last_results[{idx}] AKA P{idx+6}shape : {element.shape}")

            # append the newly calculated result to the list
            # results = [P2, P3, P4] + [ P6, P7] = [P2, P3, P4, P6, P7]
            results.extend(last_results)

            logger.debug(f"\n\t\t\t\tresults.extend(last_results)")
            logger.debug(f"\t\t\t\t// len(results): {len(results)}")

            for idx, element in enumerate(results):
                logger.debug(f"\t\t\t\t\tresults[{idx}].shape : {element.shape}")
            logger.debug(f"\n\n")

        elif isinstance(self.top_blocks, LastLevelMaxPool):
            logger.debug(f"\n\t\t\t\telif isinstance(self.top_blocks, LastLevelMaxpool):")

            last_results = self.top_blocks(results[-1])

            logger.debug(f"\t\t\t\tlast_results = self.top_blocks(results[-1])")

            results.extend(last_results)
            logger.debug(f"\t\t\t\tresults.extend(last_results)")

        logger.debug(f"\n\t\t\t#-----------------------------------")
        logger.debug(f"\t\t\t# return value tuple(results: P2, P3, P4, P6, P7) info")
        logger.debug(f"\t\t\t# which fed into RPN.forward()")
        logger.debug(f"\t\t\t#-----------------------------------")

        for idx, r in enumerate(results):
            if idx < 3:
                logger.debug(f"\t\t\tresults[{idx}] = P{idx+2} of shape: {r.shape}")
            else:
                logger.debug(f"\t\t\tresults[{idx}] = P{idx+3} of shape: {r.shape}")

        logger.debug(f"\n\treturn tuple(results)\n")
        logger.debug(f"\n\n\t}} // END FPN.forward(self,x)")
        # return as a tuple (read-only)
        return tuple(results)

# The last level of max pool layer
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        logger.debug(f"\n\tLastLevelMaxPool.forward {{ \\ BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\n\t\tParam")
        logger.debug(f"\n\t\t\tx: {x}")

        logger.debug(f"\n\treturn [F.max_pool2d(x, 1, 2, 0)]")
        logger.debug(f"\n\t}} // END LastLevelMaxPool.forward")

        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):

        logger.debug(f"\t\t# =================================")
        logger.debug(f"\t\t# 1-1-2-1 FPN.LastLevelP6P7 build")
        logger.debug(f"\t\t# =================================\n")

        logger.debug(f"\t\t\tLastLevelP6P7.__init__(self, in_channels, out_channels) {{ //BEGIN")
        logger.debug(f"\t\t\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t\t\t> Param:")
        logger.debug(f"\t\t\t\t\t>in_channels: {in_channels}")
        logger.debug(f"\t\t\t\t\t>out_channels: {out_channels}\n")

        logger.debug(f"\t\t\t\tsuper(LastLevelP6P7, self).__init__()")

        super(LastLevelP6P7, self).__init__()

        logger.debug(f"\t\t\t\tself.p6 = nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, 3, 2, 1)")
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        logger.debug(f"\t\t\t\t// self.p6: {self.p6}\n")

        logger.debug(f"\t\t\t\tself.p7 = nn.Conv2d(out_channels={out_channels}, out_channels={out_channels}, 3, 2, 1)")
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        logger.debug(f"\t\t\t\t// self.p7: {self.p7}\n")

        logger.debug(f"\t\t\t\tfor module in [self.p6, self.p7] {{")

        for module in [self.p6, self.p7]:
            logger.debug(f"\t\t\t\t\tmodule={module}")
            logger.debug(f"\t\t\t\t\tnn.init.kaiming_uniform_(module.weight=module.weight, a=1)\n")
            nn.init.kaiming_uniform_(module.weight, a=1)

            logger.debug("\t\t\t\t\tnn.init.constant_(module.bias=module.bias, 0)\n")
            nn.init.constant_(module.bias, 0)

        logger.debug(f"\t\t\t\t}} // END for module in [self.p6, self.p7]\n")

        self.use_P5 = in_channels == out_channels
        logger.debug(f"\t\t\t\tself.use_P5 = in_channels == out_channels")
        logger.debug(f"\t\t\t\t\t// self.use_p5: {self.use_P5}\n")
        logger.debug(f"\t\t\t\t}} // END LastLevelP6P7.__init__(self, in_channels, out_channels)\n\n")


    def forward(self, c5, p5):

        logger.debug(f"\t\t# =================================")
        logger.debug(f"\t\t# 2-3-1-3 FPN.LastLevelP6P7 Forward")
        logger.debug(f"\t\t# =================================\n")

        logger.debug(f"\n\t\t\tLastLevelP6P7.forward(self, c5, p5) {{ // BEGIN")
        logger.debug(f"\t\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t\t\t//Param:")
        logger.debug(f"\t\t\t\t\t//c5.shape: {c5.shape}")
        logger.debug(f"\t\t\t\t\t//p5.shape: {p5.shape}\n")

        logger.debug(f"\t\t\t\t// self.use_P5: {self.use_P5}")
        x = p5 if self.use_P5 else c5
        logger.debug(f"\t\t\t\tx = p5 if self.use_P5 else c5")

        if (self.use_P5):
            logger.debug("\t\t\t\tx=p5\n")
        else:
            logger.debug("\t\t\t\tx=c5\n")
        logger.debug(f"\t\t\t\t// x.shape = {x.shape}\n")

        p6 = self.p6(x)

        logger.debug(f"\t\t\t\t// self.p6: {self.p6}")
        logger.debug(f"\t\t\t\t// x.shape: {x.shape}")
        logger.debug(f"\t\t\t\tp6 = self.p6(x)")
        logger.debug(f"\t\t\t\t// p6.shape: {p6.shape}\n")

        file_path = f"./npy_save/P6"
        arr = p6.cpu().numpy()
        np.save(file_path, arr)
        logger.debug(f"\t\t\t\t# LastLevelP6P7::forward(), P6 of shape {arr.shape} saved into {file_path}.npy\n\n")

        p7 = self.p7(F.relu(p6))
        logger.debug(f"\t\t\t\t// self.p7: {self.p7}")
        logger.debug(f"\t\t\t\tp7 = self.p7(F.relu(p6))")
        logger.debug(f"\t\t\t\t// p7.shape: {p7.shape}\n")

        file_path = f"./npy_save/P7"
        arr = p7.cpu().numpy()
        np.save(file_path, arr)
        logger.debug(f"\t\t\t# LastLevelP6P7::forward(), P7 of shape {arr.shape} saved into {file_path}.npy\n\n")

        logger.debug(f"\t\t\t\t# ------------------")
        logger.debug(f"\t\t\t\t# return value (P6, P7) info")
        logger.debug(f"\t\t\t\t# which is appended into FPN results")
        logger.debug(f"\t\t\t\t# ------------------")
        logger.debug(f"\t\t\t\tP6 of shape {p6.shape}")
        logger.debug(f"\t\t\t\tP7 of shape {p7.shape}\n")

        logger.debug(f"\t\t\t\treturns [p6, p7]\n")
        logger.debug(f"\t\t\t}} // END LastLevelP6P7.forward(self, c5, p5)\n\n")
        return [p6, p7]
