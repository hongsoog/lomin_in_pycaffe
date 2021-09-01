# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division

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
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
    logger.debug(f"\n\tto_image_list(tensors, size_divisible={size_divisible}) ====== BEGIN")

    """
    tensors can be an ImageList, a torch.Tensor or an iterable of Tensors.
    It can't be a numpy array. When tensors is an iterable of Tensors,
    it pads the Tensors with zeros so that they have the same shape
    """

    # if parameter tensors is a torch.Tensor instance, make it to list
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        logger.debug(f"\t\tif isinstance(tensors, ImageList):")
        logger.debug(f"\t\treturn tensors")
        logger.debug(f"\tto_image_list(tensors, size_divisible={size_divisible}) ====== END\n")
        return tensors

    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        # logger.debug(f"in to_image_list 2, image_sizes: {image_sizes}")
        return ImageList(tensors, image_sizes)

    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input size
        if size_divisible > 0:
            import math

            stride = size_divisible    # size_divisible == 32
            max_size = list(max_size)  # ( C, H, W ) = (3,  480, 561 )
            # make H, W as multiple of 32 for 32 stride
            # (C, H', W') = (3, 480, 576)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        # batch_shape = (1, C, H', W') = (1, 3, 480, 576)
        batch_shape = (len(tensors),) + max_size

        # make batched_imgs by adding axis
        # bached_iamgs.shape = (1, 3, 480, 576) with all piex value is zero
        batched_imgs = tensors[0].new(*batch_shape).zero_()

        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]

        # debug (kimkk)
        logger.debug(f"\t\ttype(batched_imgs): {type(batched_imgs)}")
        #logger.debug(f"in to_image_list 2, batched_imgs: {batched_imgs}")
        logger.debug(f"\t\tbatched_imgs.shape: {batched_imgs.shape}")
        logger.debug(f"\t\timage_sizes: {image_sizes}")

        logger.debug(f"\t\treturn ImageList(batched_imgs, image_sizes)")
        logger.debug(f"\tto_image_list(tensors, size_divisible={size_divisible}) ====== END\n")
        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))
