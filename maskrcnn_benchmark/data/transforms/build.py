# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

def get_resizer(cfg):
    # resize_mode
    # - detection model: "keep_ratio"
    # - recognition model: "horizontal_padding"
    resize_mode = cfg.INPUT.RESIZE_MODE

    # target_interpolation
    #  - detection model: "bilinear"
    #  - recognition model: "bilinear"
    target_interpolation = cfg.INPUT.TARGET_INTERPOLATION

    # min_sizes
    # - detection model: 480
    # - recognition model: 800
    min_sizes = ()

    # max_sizes
    # - detection model: 640
    # - recognition model: 1333
    max_size = -1

    # fixed_size
    # - detection model = (-1, -1)
    # - recognition model = (128, 32) in format of (width, height)
    fixed_size = (-1, -1)

    # for recognition model
    if resize_mode == 'horizontal_padding':
        # fixed_size: (128,32)
        fixed_size = cfg.INPUT.FIXED_SIZE

    # for detection model
    elif resize_mode == 'keep_ratio':
        # minsizes: 480
        min_sizes = cfg.INPUT.MIN_SIZE_TEST
        min_sizes = min_sizes if isinstance(min_sizes, (tuple, list)) else (min_sizes,)
        if len(min_sizes) == 2:
            min_sizes = list(range(min_sizes[0], min_sizes[1] + 1))
        # max_size: 640
        max_size = cfg.INPUT.MAX_SIZE_TEST

    # Resize() defined in maskrcnn_benchmark/data/transforms/transforms.py
    return T.Resize(
        mode=resize_mode,
        min_sizes=min_sizes,
        max_size=max_size,
        fixed_size=fixed_size,
        target_interpolation=target_interpolation
    )

def build_transforms(cfg, is_recognition=False):
    resize_transform = get_resizer(cfg)
    normalize_transform = T.Normalize(
            # mean
            # - detection model: [1002.9801, 115.9465, 122.7717]
            # - recognition model: [1.0, 1.0, 1.0]
            mean=cfg.INPUT.PIXEL_MEAN,

            # std
            # - detection/recognition model: [1.0, 1.0, 1.0]
            std=cfg.INPUT.PIXEL_STD,

            # to_bgr255
            # - detection model: True
            # - recognition model: False
            to_bgr255=cfg.INPUT.TO_BGR255,

            # to_n1p1
            # - detection model: False
            # - recognition model: True
            to_n1p1=cfg.INPUT.TO_N1P1
    )

    transforms = list()
    post_transforms = list()

    # detection model :
    #  - transform = []
    #  - post_transforms = []
    # recognition model:
    # - transforms = []
    # - post_transforms  = []
    if is_recognition:
        # recognition model:
        # - post_transforms = [resize_transforms]
        post_transforms.append(resize_transform)
    else:
        # detection model:
        # transforms = [resize_transform]
        transforms.append(resize_transform)

    # detection model :
    #  - transform = [resize_transform]
    #  - post_transforms = []
    # recognition model:
    # - transforms = []
    # - post_transforms  = [resize_transform]

    post_transforms += [T.ToTensor(), normalize_transform]

    # detection model
    # - transforms = [resize_transform]
    # - post_transforms = [T.ToTensor(), normalize_transform]
    # recognition model
    # - transforms = []
    # - post_transforms  = [resize_transform, T.ToTensor(),  normalize_transform]
    transforms += post_transforms

    # detection model
    # - transforms = [resize_transform,  T.ToTensor(), normalize_transform]
    # recognition model
    # - transforms = [resize_transform, T.ToTensor(), normalize_transform]

    return T.Compose(transforms)

