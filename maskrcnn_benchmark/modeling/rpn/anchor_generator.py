# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import inspect
import math

import numpy as np
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

# for model debugging log
import logging
from model_log import  logger

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        logger.debug(f"BufferList.__init__(slef, buffers=None) {{ // BEGIN")
        logger.debug(f"// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"Params")
        logger.debug(f"\tlen(buffers): {len(buffers)}")

        super(BufferList, self).__init__()
        logger.debug(f"super(BufferList, self).__init__()")

        if buffers is not None:
            logger.debug(f"if buffers is not None:")
            self.extend(buffers)
            logger.debug(f"self.extend(buffers)")
            logger.debug(f"\tlen(buffers): {len(buffers)}")

        logger.debug(f"}} // END BufferList.__init__(slef, buffers=None)")

    def extend(self, buffers):
        logger.debug(f"BufferList.extend(self, buffers) {{ // BEGIN")
        logger.debug(f"// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"Params")
        logger.debug(f"\tlen(buffers): {len(buffers)}")

        offset = len(self)
        logger.debug(f"\toffset = len(self)")
        logger.debug(f"\t>> offset: {offset}")

        logger.debug(f"for i, buffer in enumerate(buffers) {{ // BEGIN")
        for i, buffer in enumerate(buffers):

            logger.debug(f"i: {i}")
            logger.debug(f"buffer.shape: {buffer.shape}")

            logger.debug(f"self.register_buffer(str(offset + i), buffer)")
            self.register_buffer(str(offset + i), buffer)

        logger.debug(f"}} // END for i, buffer in enumerate(buffers)")
        logger.debug(f"self: {self}")
        logger.debug(f"return self")
        logger.debug(f"}} // END BufferList.extend(self, buffers)")
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors

    source of explanation ://blog.csdn.net/zxfhahaha/article/details/103290259
    AnchorGenerator calculates the coordinates of all anchors on the original image according to the size of
    the original image and the size of the feature map.

    First, through the RPNModule class,
        ```
        anchor_generator = make_anchor_generator(cfg) #生成anchorswe first generate anchors
        ```
    from cfg.RETINANET:ANCHOR_SIZES,
   ANCHOR_SIZES: (32, 64, 128, 256, 512) #anchor area size
   ANCHOR_STRIDE: (4, 8, 16, 32, 64)     #
   ASPECT_RATIOS: (0.5, 1.0, 2.0)        #In fact, it is the multiple of downsampling

   Then call the AnchorGenerator.forward() function to generate anchors, specifically for each stride and size pair
   (4,32),(8,64),(16,128),(32,256),(64,512) to generate cell_anchors(x1,y1, x2,y2),
    """

    # AnchorGenerator:__init_ constructor
    def __init__( self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0), anchor_strides=(8, 16, 32), straddle_thresh=0, ):

        """
        sizes:  anchor area size, default: (128, 256, 512)
            QUESTION: where does this argument comes from ??
            ANSWER: make_anchor_generator_retinanet(config):

            (( 32.0, 40.31747359663594, 50.79683366298238),
             ( 64.0, 80.63494719327188, 101.59366732596476),
             (128.0, 161.26989438654377, 203.18733465192952),
             (256.0, 322.53978877308754, 406.37466930385904),
             (512.0, 645.0795775461751,  812.7493386077181))

        aspect_ratios: used for multiple of resizing anchor, default: (0.5, 1.0, 2.0)
             (0.5, 1.0, 2.0)
        anchor_strides:
             (8, 16, 32, 64, 128)
        straddle_thresh:
        """
        if logger.level == logging.DEBUG:
            logger.debug(f"\t\tAnchorGenerator.__init__(sizes, aspect_ratios, anchor_strides, straddle_thresh) {{ //BEGIN")
            logger.debug(f"\t\t// defined in {inspect.getfile(inspect.currentframe())}")
            logger.debug(f"\t\t\tParams")
            logger.debug(f"\t\t\t\tsizes: {sizes}")
            logger.debug(f"\t\t\t\taspect_ratios: {aspect_ratios}")
            logger.debug(f"\t\t\t\tanchor_strides: {anchor_strides}")
            logger.debug(f"\t\t\t\tstraddle_thresh: {straddle_thresh}")


        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tif len(anchor_strides) ==1 ")
                logger.debug(f"\t\t\tanchor_stride = anchor_strides[0]")
                logger.debug(f"\t\t\t cell_anchors = [ generate_anchors(anchor_stride, sizes, aspect_ratios).float() ]")

            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\telse: i.e, len(anchor_strides) !=1")
                logger.debug(f"\t\t\tanchor_stride = anchor_strides[0]")
                logger.debug(f"\t\t\tlen(anchor_strides):{len(anchor_strides)}, len(size): {len(sizes)}")

            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\telse: i.e, len(anchor_strides) == len(sizes)")
                logger.debug(f"\t\tcell_anchors = [ generate_anchors( anchor_stride,")
                logger.debug(f"\t\t                 size if isinstance(size, (tuple, list)) else (size,), ")
                logger.debug(f"\t\t                 aspect_ratios).float()")
                logger.debug(f"\t\tfor anchor_stride, size in zip(anchor_strides, sizes)")

            cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                ).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]

        if logger.level == logging.DEBUG:
            logger.debug(f"cell_anchors: {cell_anchors}")
        """
        cell_anchors: 
        For each pair of anchor_stride (8, 16, 32) and anchor_size (128, 256, 512), three anchors are generated in the ratio of 1:1 1:2 2:1, in the form of (x1,y1,x2,y2)
        hence total number of anchors are 3x3x3 = 27
        
                    [tensor([[-22., -10.,  25.,  13.],   # (x1, y1, x2, y2)  aspect ratio 1:1
                             [-14., -14.,  17.,  17.],   # (x1, y1, x2, y2)  aspect ratio 1:2
                             [-10., -22.,  13.,  25.]]), # (x1, y1, x2, y2)  aspect ratio 2:1
                             
                     tensor([[-40., -20.,  47.,  27.],
                             [-28., -28.,  35.,  35.],
                             [-20., -44.,  27.,  51.]]), 
                             
                     tensor([[-84., -40.,  99.,  55.],
                             [-56., -56.,  71.,  71.],
                             [-36., -80.,  51.,  95.]]), 
                             
                     tensor([[-164.,  -72.,  195.,  103.],
                             [-112., -112.,  143.,  143.],
                             [ -76., -168.,  107.,  199.]]), 
                             
                     tensor([[-332., -152.,  395.,  215.],
                             [-224., -224.,  287.,  287.],
                             [-148., -328.,  211.,  391.]])]
        """
        self.strides = anchor_strides
        logger.debug(f"\tself.strides = anchor_strides")
        logger.debug(f"\t>> self.strides: {self.strides}")

        logger.debug(f"\tself.cell_anchors = BufferList(cell_anchors) // CALL")
        self.cell_anchors = BufferList(cell_anchors)
        logger.debug(f"\tself.cell_anchors = BufferList(cell_anchors) // RETURNED")
        logger.debug(f"\t>> self.cell_anchors: {self.cell_anchors}")

        self.straddle_thresh = straddle_thresh
        logger.debug(f"\tself.straddle_thresh = straddle_thresh")
        logger.debug(f"\t>> self.straddle_thresh: {self.straddle_thresh}")

        logger.debug(f"\t\t}} // END AnchorGenerator.__init__(sizes, aspect_ratios, anchor_strides, straddle_thresh)")

    # AnchorGenerator:num_anchors_per_location() method
    def num_anchors_per_location(self):
        if logger.level == logging.DEBUG:
            logger.debug(f"AnchorGenerator.num_anchors_per_location() {{ //BEGIN")

        import pdb; pdb.set_trace()

        if logger.level == logging.DEBUG:
            logger.debut(f"\t\t\treturn [len(cell_anchors) for cell_anchors in self.cell_anchors]")
            logger.debug(f"\t\t}} // END AnchorGenerator.num_anchors_per_location()\n")

        return [len(cell_anchors) for cell_anchors in self.cell_anchors]


    # AnchorGenerator:grid_sizes() method
    def grid_anchors(self, grid_sizes):
        if logger.level == logging.DEBUG:
            logger.debug(f"\t\tAnchorGenerator.grid_anchors(grid_sizes) {{ // BEGIN")
            logger.debug(f"\t\t\tParam:")
            logger.debug(f"\t\t\tgrid_sizes: {grid_sizes}")

        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        if logger.level == logging.DEBUG:
            logger.debug(f"return anchors")
            logger.debug(f"\t\t}} // END AnchorGenerator.grid_anchors(grid_sizes)\n")
        return anchors

    def add_visibility_to(self, boxlist):
        if logger.level == logging.DEBUG:
            logger.debug(f"AnchorGenerator.add_visibitity_to(boxlist) {{ // BEGIN")
            logger.debug(f"// defined in {inspect.getfile(inspect.currentframe())}")

        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.bool, device=device)
        boxlist.add_field("visibility", inds_inside)

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t}} // END AnchorGenerator.add_visibitity_to(boxlist)\n")

    def forward(self, image_list, feature_maps):
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\tAnchorGenerator.forward(image_list, feature_maps) {{ //BEGIN")
            logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}")
            logger.debug(f"\t\tParams:")
            logger.debug(f"\t\t\timage_list:")
            logger.debug(f"\t\t\t\tlen(image_list.image_sizes): {len(image_list.image_sizes)}")
            logger.debug(f"\t\t\t\timage_list.image_sizes[0]: {image_list.image_sizes[0]}")
            logger.debug(f"\t\t\t\tlen(image_list.tensors): {len(image_list.tensors)}")
            logger.debug(f"\t\t\t\timage_list.tensors[0].shape: {image_list.tensors[0].shape}")
            logger.debug(f"\t\t\tfeature_maps:")
            for idx, f in enumerate(feature_maps):
                logger.debug(f"\t\t\t\tfeature_maps[{idx}].shape: {f.shape}")

            logger.debug(f"\ngrid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]")

        # extract h and w from feature maps of shape (b, c, h, w)
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]

        if logger.level == logging.DEBUG:
            logger.debug(f"anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)")

        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        if logger.level == logging.DEBUG:
            logger.debug(f"anchors = []")

        anchors = []

        if logger.level == logging.DEBUG:
            logger.debug(f"for i, (image_height, image_width) in enumerate(image_list.image_sizes) {{\n")

        for i, (image_height, image_width) in enumerate(image_list.image_sizes):

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tanchors_in_image = []\n")

            anchors_in_image = []

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tfor anchors_per_feature_map in anchors_over_all_feature_maps {{\n")

            for anchors_per_feature_map in anchors_over_all_feature_maps:

                if logger.level == logging.DEBUG:
                    logger.debug(f'\t\t========================')
                    logger.debug(f'\t\tanchors_per_feature_map.shape: {anchors_per_feature_map.shape}')
                    logger.debug(f'\t\t========================')

                    logger.debug(f'\t\tboxlist = BoxList( anchors_per_feature_map, (image_width, image_height), mode="xyxy" )')

                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )

                if logger.level == logging.DEBUG:
                    logger.debug(f"\t\tboxlist:\n\t\t\t{boxlist}\n")
                    logger.debug(f"\t\tself.add_visibility_to(boxlist)\n")

                self.add_visibility_to(boxlist)

                if logger.level == logging.DEBUG:
                    logger.debug(f"\t\tboxlist:\n\t\t\t{boxlist}\n")
                    logger.debug(f"\t\tanchors_in_image.append(boxlist)\n")

                anchors_in_image.append(boxlist)

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t}} // END for anchors_per_feature_map in anchors_over_all_feature_maps\n")

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\tanchors_in_image:\n\t\t\t{anchors_in_image}\n")
                logger.debug(f"\t\tanchors.append(anchors_in_image)\n")

            anchors.append(anchors_in_image)
        if logger.level == logging.DEBUG:
            logger.debug(f"}} // END for i, (image_height, image_width) in enumerate(image_list.image_sizes)\n")

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\tanchors:\n\t\t\t{anchors}\n")
            logger.debug(f"return anchors")
            logger.debug(f"\t}} // END AnchorGenerator.forward(image_list, feature_maps)")
        return anchors


def make_anchor_generator(config):


    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    if logger.level == logging.DEBUG:
        logger.debug(f"\n\n\tmake_anchor_generator(config) {{ // BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\tanchor_sizes: {anchor_sizes}")
        logger.debug(f"\t\taspect_ratios: {aspect_ratios}")
        logger.debug(f"\t\tanchor_stride: {anchor_stride}")
        logger.debug(f"\t\tstraddle_thresh: {straddle_thresh}")
        logger.debug(f"\t\tconfig.MODEL.RPN_USE_FPN: {config.MODEL.RPN.USE_FPN}")

    if config.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len( anchor_sizes ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"

    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"

        anchor_generator = AnchorGenerator( anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh )
        if logger.level == logging.DEBUG:
            logger.debug(f"\t\tanchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh)")
            logger.debug(f"\t\tanchor_generator: {anchor_generator}")

    if logger.level == logging.DEBUG:
        logger.debug(f"return anchor_generator")
        logger.debug(f"\t}} //make_anchor_generator(config) END\n")

    return anchor_generator


def make_anchor_generator_retinanet(config):
    anchor_sizes = config.MODEL.RETINANET.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RETINANET.ASPECT_RATIOS
    anchor_strides = config.MODEL.RETINANET.ANCHOR_STRIDES
    straddle_thresh = config.MODEL.RETINANET.STRADDLE_THRESH
    octave = config.MODEL.RETINANET.OCTAVE
    scales_per_octave = config.MODEL.RETINANET.SCALES_PER_OCTAVE

    if logger.level == logging.DEBUG:
        logger.debug(f"\n\t\tmake_anchor_generator_retinanet(config) {{ // BEGIN")
        logger.debug(f"// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\tconfig params")
        logger.debug(f"\t\t\tanchor_sizes: {anchor_sizes}")
        logger.debug(f"\t\t\taspect_ratios: {aspect_ratios}")
        logger.debug(f"\t\t\tanchor_strides: {anchor_strides}")
        logger.debug(f"\t\t\tstraddle_thresh: {straddle_thresh}")
        logger.debug(f"\t\t\toctave: {octave}")
        logger.debug(f"\t\t\tscales_per_octave: {scales_per_octave}")

    assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\tnew_anchor_sizes = []")

    new_anchor_sizes = []

    if logger.level == logging.DEBUG:
        logger.debug(f"\n\t\tfor size in anchor_sizes {{")

    for size in anchor_sizes:  # loop over anchor_sizes

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\t-----------")
            logger.debug(f"\t\t\tsize: {size}")
            logger.debug(f"\t\t\t-----------")
            logger.debug(f"\t\t\tper_layer_anchor_sizes = []")

        per_layer_anchor_sizes = []

        if logger.level == logging.DEBUG:
            logger.debug(f"\n\t\t\tfor scale_per_octave in range(scales_per_octave={scales_per_octave}) {{ // BEGIN")

        for scale_per_octave in range(scales_per_octave):  # loop over scales_per_octave

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t\t\t------------------------")
                logger.debug(f"\t\t\t\tscale_per_octave : {scale_per_octave}")
                logger.debug(f"\t\t\t\t------------------------")
                logger.debug(f"\t\t\t\t\toctave : {octave}\n")
                logger.debug(f"\t\t\t\toctave_scale = octave ** (scale_per_octave / float(scales_per_octave))\n")

            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t\t\t\toctave_scale: {octave_scale}")
                logger.debug(f"\t\t\t\t\tsize: {size}\n")
                logger.debug(f"\t\t\t\tper_layer_anchor_sizes.append(octave_scale * size)")

            per_layer_anchor_sizes.append(octave_scale * size)

            if logger.level == logging.DEBUG:
                logger.debug(f"\t\t\t\t\tper_layer_anchor_sizes: {per_layer_anchor_sizes}\n")

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\t}} // EDN for scale_per_octave in range(scales_per_octave)\n\n")

        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t\tnew_anchor_sizes.append(tuple(per_layer_anchor_sizes))")
            logger.debug(f"\t\t\tnew_anchor_sizes: {new_anchor_sizes}")

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t}} // END for size in anchor_sizes\n")

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\tnew_anchor_sizes:\n\t\t\t{new_anchor_sizes}")
        logger.debug(f"\t\taspect_ratios:\n\t\t\t{aspect_ratios}")
        logger.debug(f"\t\tanchor_strides:\n\t\t\t{anchor_strides}")
        logger.debug(f"\t\tstraddle_thresh:\n\t\t\t{straddle_thresh}")
        logger.debug(f"\t\tanchor_generator = AnchorGenerator( tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh )")

    anchor_generator = AnchorGenerator(
        tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh
    )

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\tanchor_generator = {anchor_generator}")
        logger.debug(f"\n\t\treturn anchor_generator")
        logger.debug(f"\t\t}} // make_anchor_generator_retinanet(config) END\n")

    return anchor_generator

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\tgenerate_anchors(stride, sizes, aspect_ratios) {{ //BEGIN")
        logger.debug(f"\t\t\t//defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\t\t\t\tParams:")
        logger.debug(f"\t\t\t\t\t\tstride: {stride}")
        logger.debug(f"\t\t\t\t\t\tsizes: {sizes}")
        logger.debug(f"\t\t\t\t\t\taspect_ratios: {aspect_ratios}")


    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\t\treturn _generate_anchors(stride,")
        logger.debug(f"\t\t\t\t\t\t     np.array(sizes, dtype=np.float) / stride,")
        logger.debug(f"\t\t\t\t\t\t     np.array(aspect_ratios, dtype=np.float),)")

    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\t\t\t_generate_anchors(base_size, scales, aspect_ratios) {{ //BEGIN")
        logger.debug(f"\t\t\t\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\t\t\t\t\t\tParams:")
        logger.debug(f"\t\t\t\t\t\t\t\tbase_size: {base_size}")
        logger.debug(f"\t\t\t\t\t\t\t\tscales: {scales}")
        logger.debug(f"\t\t\t\t\t\t\t\taspect_ratios: {aspect_ratios}")


    if logger.level == logging.DEBUG:
        logger.debug(f"anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1")

    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1

    if logger.level == logging.DEBUG:
        logger.debug(f"anchor: {anchor}")
        logger.debug(f"anchors = _ratio_enum(anchor, aspect_ratios)")

    anchors = _ratio_enum(anchor, aspect_ratios)

    if logger.level == logging.DEBUG:
        logger.debug(f"anchors: {anchors}")
        logger.debug(f"anchors = np.vstack(")
        logger.debug(f"[_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]")
        logger.debug(f")")

    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )

    if logger.level == logging.DEBUG:
        logger.debug(f"anchors: {anchors}")

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\t\t\t\treturn torch.from_numpy(anchors)")
        logger.debug(f"\t\t\t\t\t\t}} // END _generate_anchors(base_size, scales, apect_ratios) END")
        logger.debug(f"\t\t\t\t\t}} // END generate_anchors(stride, sizes, aspect_ratios)")

    return torch.from_numpy(anchors)

def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window).
         (a[0], a[1]) = (X1, Y1)
             P1
             X--------------------------+  ---
             |<------------------------>|   |
             |      w = a[2] - a[0]     |   |  h = a[3] - a[1]
             |                          |   |
             |                          |   |
             |                          |   |
             +--------------------------X  ---
                                         P2
                                         (a[2], a[3]) = (X2, Y2)

    """

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\t_whctrs(anchors) {{ //BEGIN")
        logger.debug(f"\t\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\t\t\t\tParam:")
        logger.debug(f"\t\t\t\t\t\tanchor: {anchor}")

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)

    logger.debug(f"\t\t\t\t\t\tw = anchor[2] - anchor[0] + 1")
    logger.debug(f"\t\t\t\t\t\tw: {w}")
    logger.debug(f"\t\t\t\t\t\th = anchor[3] - anchor[1] + 1")
    logger.debug(f"\t\t\t\t\t\th: {h}")
    logger.debug(f"\t\t\t\t\t\tx_ctr = anchor[0] + 0.5 * (w - 1)")
    logger.debug(f"\t\t\t\t\t\tx_ctr: {x_ctr}")
    logger.debug(f"\t\t\t\t\t\ty_ctr = anchor[1] + 0.5 * (h - 1)")
    logger.debug(f"\t\t\t\t\t\ty_ctr: {y_ctr}")

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\t\t\treturn w, h, x_ctr, y_ctr")
        logger.debug(f"\t\t\t\t\t}} // END _whctrs(anchors)")

    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t_mkanchors(ws, hs, x_ctr, y_ctr) {{ // BEGIN")
        logger.debug(f"\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\t\t\tParam:")
        logger.debug(f"\t\t\t\t\tws: {ws}")
        logger.debug(f"\t\t\t\t\ths: {hs}")
        logger.debug(f"\t\t\t\t\tx_ctr: {x_ctr}")
        logger.debug(f"\t\t\t\t\ty_ctr: {y_ctr}")

    ws = ws[:, np.newaxis]

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\tws = ws[:, np.newaxis]")
        logger.debug(f"\t\t\t\t\tws: {ws}")

    hs = hs[:, np.newaxis]

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\ths = hs[:, np.newaxis]")
        logger.debug(f"\t\t\t\t\ths: {hs}")
        logger.debug(f"\t\t\t\tanchors = np.hstack(")
        logger.debug(f"\t\t\t\t    (")
        logger.debug(f"\t\t\t\t        x_ctr - 0.5 * (ws - 1),")
        logger.debug(f"\t\t\t\t        y_ctr - 0.5 * (hs - 1),")
        logger.debug(f"\t\t\t\t        x_ctr + 0.5 * (ws - 1),")
        logger.debug(f"\t\t\t\t        y_ctr + 0.5 * (hs - 1),")
        logger.debug(f"\t\t\t\t    )")
        logger.debug(f"\t\t\t\t)")

    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\tanchors: {anchors}")
        logger.debug(f"\t\t\t\treturn anchors")
        logger.debug(f"\t\t\t}} // END _mkanchors(ws, hs, x_ctr, y_ctr)")

    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\t_ratio_enum(anchor, ratios) {{ //BEGIN")
        logger.debug(f"\t\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\t\t\t\tParam:")
        logger.debug(f"\t\t\t\t\t\tanchor: {anchor}")
        logger.debug(f"\t\t\t\t\t\tratios: {ratios}")

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    logger.debug(f"\t\t\t\t\tw, h, x_ctr, y_ctr = _whctrs(anchor)")
    logger.debug(f"\t\t\t\t\tw: {w}, h: {h}, x_ctr: {x_ctr}, y_ctr: {y_ctr}")

    size = w * h
    logger.debug(f"\t\t\t\t\tsize = w * h")
    logger.debug(f"\t\t\t\t\tsize: {size}")

    size_ratios = size / ratios
    logger.debug(f"\t\t\t\t\tsize_ratios = size / ratios")
    logger.debug(f"\t\t\t\t\tsize_ratios: {size_ratios}")


    ws = np.round(np.sqrt(size_ratios))
    logger.debug(f"\t\t\t\t\tws = np.round(np.sqrt(size_ratios))")
    logger.debug(f"\t\t\t\t\tws: {ws}")

    hs = np.round(ws * ratios)
    logger.debug(f"\t\t\t\t\ths = np.round(ws * ratios)")
    logger.debug(f"\t\t\t\t\ths: {hs}")

    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    logger.debug(f"\t\t\t\t\tanchors = _mkanchors(ws, hs, x_ctr, y_ctr)")

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\t\tanchors: {anchors}")
        logger.debug(f"\t\t\t\t\treturn anchors")
        logger.debug(f"\t\t\t\t}} // END _ratio_enum(anchor, ratios)")

    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\t\t_scale_enum(anchor, scales) {{ //BEGIN")
        logger.debug(f"\t\t\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\t\t\t\tParam:")
        logger.debug(f"\t\t\t\t\t\tanchor: {anchor}")
        logger.debug(f"\t\t\t\t\t\tscales: {scales}")

    logger.debug(f"\t\t\t\t\tw, h, x_ctr, y_ctr = _whctrs(anchor)")

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    logger.debug(f"\t\t\t\t\tw: {w}, h: {h}, x_ctr: {x_ctr}, y_ctr: {y_ctr}")

    logger.debug(f"\t\t\t\t\tws = w * scales")
    ws = w * scales
    logger.debug(f"\t\t\t\t\tws: {ws}")

    logger.debug(f"\t\t\t\t\ths = h * scales")
    hs = h * scales
    logger.debug(f"\t\t\t\t\ths: {hs}")

    logger.debug(f"\t\t\t\t\tanchors = _mkanchors(ws, hs, x_ctr, y_ctr)")
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\t\t\tanchors: {anchors}")
        logger.debug(f"\t\t\t\t}} // END _scale_enum(anchor, scales)")

    return anchors
