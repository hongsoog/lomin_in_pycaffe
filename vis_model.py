# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from torchviz import make_dot


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Model Visualisation")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output_path",
        default="pytorchviz_output.dot",
        type=str,
        help="Save path for pytorchviz output [pytorchviz_output.dot]",
    )
    parser.add_argument(
        "--output_format",
        default="pdf",
        type=str,
        help="Output format for pytorchviz output [pdf]",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


def make_dummy_image(size):
    return torch.randn(size).requires_grad_(True)


def make_dummy_masks(boxlist):
    xmin, ymin, xmax, ymax = boxlist.convert('xyxy')._split_into_xyxy()
    polygons = []
    for (xmin_per_box, ymin_per_box, xmax_per_box, ymax_per_box) in zip(xmin, ymin, xmax, ymax):
        polygon_per_box = [[xmin_per_box, ymin_per_box,
                            xmin_per_box, ymax_per_box,
                            xmax_per_box, ymax_per_box,
                            xmax_per_box, ymin_per_box]]
        polygons.append(polygon_per_box)
    return SegmentationMask(polygons, boxlist.size)


def make_dummy_target(size, num_regions, num_classes, mode='xyxy', mask_on=False):
    if mode not in ['xyxy', 'xywh']:
        raise ValueError('mode expects xyxy or xywh, but gets {}'.format(mode))
    h = size[0]
    w = size[1]
    TO_REMOVE = 1
    dummy_x = torch.rand(num_regions, 1) * (w - TO_REMOVE)
    dummy_y = torch.rand(num_regions, 1) * (h - TO_REMOVE)
    dummy_w = torch.rand(num_regions, 1) * (w - TO_REMOVE)
    dummy_h = torch.rand(num_regions, 1) * (h - TO_REMOVE)

    dummy_x = torch.round(dummy_x)
    dummy_y = torch.round(dummy_y)
    dummy_w = torch.round(dummy_w)
    dummy_h = torch.round(dummy_h)

    dummy_x_max = (dummy_x + dummy_w - TO_REMOVE).clamp(max=(w - TO_REMOVE))
    dummy_y_max = (dummy_y + dummy_h - TO_REMOVE).clamp(max=(h - TO_REMOVE))
    dummy_w = dummy_x_max - dummy_x + TO_REMOVE
    dummy_h = dummy_y_max - dummy_y + TO_REMOVE
    if mode == 'xyxy':
        bbox = torch.cat((dummy_x, dummy_y, dummy_x_max, dummy_y_max), dim=-1)
    elif mode == 'xywh':
        bbox = torch.cat((dummy_x, dummy_y, dummy_w, dummy_h), dim=-1)
    else:
        raise RuntimeError('Should not be here')
    boxlist = BoxList(bbox, (h, w), mode=mode)

    # Make dummy_labels
    dummy_labels = torch.rand(num_regions) * (num_classes - TO_REMOVE)
    dummy_labels = torch.round(dummy_labels)
    boxlist.add_field('labels', dummy_labels)

    # Make dummy masks if mask_on = True
    if mask_on == True:
        dummy_masks = make_dummy_masks(boxlist)
        boxlist.add_field('masks', dummy_masks)
    return boxlist


def main():
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_detection_model(cfg)

    dummy_size = (800, 1600)  # (image_height, image_width)
    RGB_CHANNEL = 3
    NUM_BBOX = 5
    mask_on = cfg.MODEL.MASK_ON
    num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

    dummy_image = make_dummy_image((1, RGB_CHANNEL, dummy_size[1], dummy_size[0]))
    dummy_target = make_dummy_target(dummy_size, NUM_BBOX, num_classes, mask_on=mask_on)
    loss_dict = model(dummy_image, [dummy_target])
    # loss_dict is a dict with the following possible key-value pairs:
    # 1) RPN losses:
    #   'loss_objectness': tensor(_loss_value_, grad_fn=<BinaryCrossEntropyWithLogitsBackward>),
    #   'loss_rpn_box_reg': tensor(_loss_value_, grad_fn=<DivBackward0>)
    # 2) Detection losses:
    #   'loss_classifier': tensor(_loss_value_, grad_fn=<NllLossBackward>),
    #   'loss_box_reg': tensor(_loss_value_, grad_fn=<DivBackward0>),
    # 3) Mask losses:
    #   'loss_mask': tensor(_loss_value_, grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
    # Some losses might not be present depending on actual cfg settings
    assert isinstance(loss_dict, dict), "loss_dict expects dict, but got {}".format(type(loss_dict))
    loss = tuple(loss_dict.values())
    graph = make_dot(loss, params=dict(list(model.named_parameters()) + [('dummy_image', dummy_image)]))
    graph.format = args.output_format
    graph.render(args.output_path)
    print('Visualisation saved at {}.{}'.format(os.path.abspath(args.output_path), args.output_format))
    print('Raw graph saved at {}'.format(os.path.abspath(args.output_path)))


if __name__ == "__main__":
    main()