# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division

import inspect
import torch

# for model debugging log
from model_log import  logger

class ImageList(object):
    """
    Structure that holds a list of images (of possibly varying sizes) as a single tensor.
    This works by padding the images to the same size, and storing in a image_sizes field
    the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        logger.debug(f"\n\tImageList.__init__(self, tensors, image_sizes) {{ // BEGIN")
        logger.debug(f"\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t// Params:")
        logger.debug(f"\t\t\t> tensors.shape: {tensors.shape}")
        logger.debug(f"\t\t\t> image_sizes: {image_sizes}\n")

        self.tensors = tensors
        self.image_sizes = image_sizes

        logger.debug(f"\t\tself.tensors = tensors")
        logger.debug(f"\t\tself.image_sizes = image_sizes")
        logger.debug(f"\t}} // END ImageList.__init__(self, tensors, image_sizes)\n")

    def to(self, *args, **kwargs):
        logger.debug(f"\n\tImageList.to(self, *args, **kwargs) {{ // BEGIN")
        logger.debug(f"\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t// Params:")
        logger.debug(f"\t\t\targs: {args}")
        logger.debug(f"\t\t\tkwargs: {kwargs}\n")

        logger.debug(f"\t\tcast_tensor = self.tensors.to(*args, **kwargs)")
        cast_tensor = self.tensors.to(*args, **kwargs)
        logger.debug(f"\t\t// cast_tensor: cast_tensor")

        logger.debug(f"\t}} // END ImageList.to(self, *args, **kwargs)\n")
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):

    """
    tensors can be an ImageList, a torch.Tensor or an iterable of Tensors.
    It can't be a numpy array. When tensors is an iterable of Tensors,
    it pads the Tensors with zeros so that they have the same shape
    """
    logger.debug(f"\n\tto_image_list(tensors, size_divisible=0) {{ // BEGIN")
    logger.debug(f"\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
    logger.debug(f"\t\t// Params:")
    logger.debug(f"\t\t\t> type(tensors): {type(tensors)}")
    logger.debug(f"\t\t\t> size_divisible: {size_divisible}\n")

    # if parameter tensors is a torch.Tensor instance, make it to list
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        logger.debug(f"\t\tif isinstance(tensors, torch.Tensor) and size_divisible > 0:")
        tensors = [tensors]
        logger.debug(f"\t\t\ttensors = [tensors]")
        logger.debug(f"\t\t\t// len(tensors]: {len(tensors)}")
        logger.debug(f"\t\t\t// tensors[0].shape: {tensors[0].shape}\n")

    if isinstance(tensors, ImageList):
        logger.debug(f"\t\tif isinstance(tensors, ImageList):")
        logger.debug(f"\t\t\treturn tensors\n")
        logger.debug(f"\n\t}} // END to_image_list(tensors, size_divisible=0)")
        return tensors

    elif isinstance(tensors, torch.Tensor):
        logger.debug(f"\t\telif isinstance(tensors, torch.Tensor):")

        # single tensor shape can be inferred
        if tensors.dim() == 3:
            logger.debug(f"\t\t\tif tensors.dim() == 3:")
            tensors = tensors[None]
            logger.debug(f"\t\t\t\ttensors = tensors[None]")
            logger.debug(f"\t\t\t\t// tensors.shape: {tensors.shape}")

        logger.debug(f"\t\t\tassert tensors.dim() == 4")
        assert tensors.dim() == 4

        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        logger.debug(f"\t\t\timage_sizes = [tensor.shape[-2:] for tensor in tensors]")
        logger.debug(f"\t\t\t// image_sizes: {image_sizes}")

        # logger.debug(f"in to_image_list 2, image_sizes: {image_sizes}")
        logger.debug(f"\t\treturn ImageList(tensors, image_sizes)")
        logger.debug(f"\n\t}} // END to_image_list(tensors, size_divisible=0)")
        return ImageList(tensors, image_sizes)

    elif isinstance(tensors, (tuple, list)):
        logger.debug(f"\t\telif isinstance(tensors, (tuple, list)):")

        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
        logger.debug(f"\t\t\tmax_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))")
        logger.debug(f"\t\t\t// max_size: {max_size}\n")

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input size
        if size_divisible > 0:
            logger.debug(f"\t\t\tif size_divisible > 0:")
            logger.debug(f"\t\t\t\timport math\n")
            import math

            stride = size_divisible    # size_divisible == 32
            logger.debug(f"\t\t\t\tstride = size_divisible")
            logger.debug(f"\t\t\t\t// stride: {stride}\n")

            max_size = list(max_size)  # ( C, H, W ) = (3,  480, 561 )
            logger.debug(f"\t\t\t\tmax_size = list(max_size)")
            logger.debug(f"\t\t\t\t// max_size: {max_size}\n")
            # make H, W as multiple of 32 for 32 stride
            # (C, H', W') = (3, 480, 576)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

            logger.debug(f"\t\t\t\tmax_size[1] = int(math.ceil(max_size[1] / stride) * stride)")
            logger.debug(f"\t\t\t\t// max_size[1]: {max_size[1]}\n")
            logger.debug(f"\t\t\t\tmax_size[2] = int(math.ceil(max_size[2] / stride) * stride)")
            logger.debug(f"\t\t\t\t// max_size[2]: {max_size[2]}\n")
            logger.debug(f"\t\t\t\tmax_size = tuple(max_size)")
            logger.debug(f"\t\t\t\tmax_size: {max_size}\n")

        # batch_shape = (1, C, H', W') = (1, 3, 480, 576)
        batch_shape = (len(tensors),) + max_size
        logger.debug(f"\t\t\tbatch_shape = (len(tensors),) + max_size")
        logger.debug(f"\t\t\tbatch_shape: {batch_shape}\n")

        # make batched_imgs by adding axis
        # bached_imags.shape = (1, 3, 480, 576) with all pixel value is zero
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        logger.debug(f"\t\t\t# make batch_imgs by adding axis with all pixel values is zero")
        logger.debug(f"\t\t\tbatched_imgs = tensors[0].new(*batch_shape).zero_()")
        logger.debug(f"\t\t\tbatched_imgs.shape: {batched_imgs.shape}\n")

        logger.debug(f"\t\t\t# overlay tensors on batch_imgs (pad)img")
        logger.debug(f"\t\t\tfor img, pad_img in zip(tensors, batched_imgs):")
        logger.debug(f"\t\t\t\tpad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)\n")
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]
        logger.debug(f"\t\t\timage_sizes = [im.shape[-2:] for im in tensors]")
        logger.debug(f"\t\t\t// image_sizes: {image_sizes}\n")

        # debug (kimkk)
        logger.debug(f"\t\t\t// type(batched_imgs): {type(batched_imgs)}\n")
        #logger.debug(f"in to_image_list 2, batched_imgs: {batched_imgs}")
        logger.debug(f"\t\t\t// batched_imgs.shape: {batched_imgs.shape}")
        logger.debug(f"\t\t\t// image_sizes: {image_sizes}\n")

        logger.debug(f"\t\t\treturn ImageList(batched_imgs, image_sizes) // CALL")
        #logger.debug(f"\n\t}} // END to_image_list(tensors, size_divisible=0)")
        return ImageList(batched_imgs, image_sizes)
    else:
        logger.debug(f"\t\telse:")
        logger.debug(f'\t\t\traise TypeError("Unsupported type for to_image_list: {{}}".format(type(tensors)))')
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))
