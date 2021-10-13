# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch

# for model debugging log
import logging
from model_log import  logger
import inspect

class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        logger.debug(f"\t\tBoxCoder.__init__(self, weights, bbox_xfrom_clip)__ {{ // BEGIN")
        logger.debug(f"\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\t\t// Params:")
        logger.debug(f"\t\t\t\tweights: {weights}")
        logger.debug(f"\t\t\t\tbbox_xform_clip: {bbox_xform_clip}")

        self.weights = weights
        logger.debug(f"\t\t\t\tself.weights = weights")

        self.bbox_xform_clip = bbox_xform_clip
        logger.debug(f"\t\t\t\tself.bbox_xform_clip = bbox_xform_clip\n")
        logger.debug(f"\t\t}} // END BoxCoder.__init__(self, weights, bbox_xfrom_clip)__\n")

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        logger.debug(f"\t\tBoxCoder.encode(self, reference_boxes, proposals)__ {{ // BEGIN")
        logger.debug(f"\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\t\t// Params:")
        logger.debug(f"\t\t\t\treference_boxes: {reference_boxes}")
        logger.debug(f"\t\t\t\tproposals: {proposals}\n")

        TO_REMOVE = 1  # TODO remove
        logger.debug(f"\t\t\tTO_REMOVE = 1  # TODO remove")

        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        logger.debug(f"\t\t\tex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE")
        logger.debug(f"\t\t\t// ex_widths: {ex_widths}")

        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        logger.debug(f"\t\t\tex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE")
        logger.debug(f"\t\t\t// ex_heights: {ex_heights}")

        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        logger.debug(f"\t\t\tex_ctr_x = proposals[:, 0] + 0.5 * ex_widths")
        logger.debug(f"\t\t\t// ex_ctr_x: {ex_ctr_x}")

        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
        logger.debug(f"\t\t\tex_ctr_y = proposals[:, 1] + 0.5 * ex_heights")
        logger.debug(f"\t\t\t// ex_ctr_y: {ex_ctr_y}")


        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        logger.debug(f"\t\t\tgt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE")
        logger.debug(f"\t\t\t// gt_widths: {gt_widths}")

        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        logger.debug(f"\t\t\tgt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE")
        logger.debug(f"\t\t\t// gt_heights: {gt_heights}")

        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        logger.debug(f"\t\t\tgt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths")
        logger.debug(f"\t\t\t// gt_ctr_x: {gt_ctr_x}")

        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights
        logger.debug(f"\t\t\tgt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights")
        logger.debug(f"\t\t\t// gt_ctr_y: {gt_ctr_y}\n")


        wx, wy, ww, wh = self.weights
        logger.debug(f"\t\t\twx, wy, ww, wh = self.weights")
        logger.debug(f"\t\t\t// wx: {wx}, wy: {wy}, ww: {ww}, wh: {wh}")

        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        logger.debug(f"\t\t\ttargets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths")
        logger.debug(f"\t\t\t// targets_dx: {targets_dx}")

        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        logger.debug(f"\t\t\ttargets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights")
        logger.debug(f"\t\t\t// targets_dy: {targets_dy}")

        targets_dw = ww * torch.log(gt_widths / ex_widths)
        logger.debug(f"\t\t\ttargets_dw = ww * torch.log(gt_widths / ex_widths)")
        logger.debug(f"\t\t\t// targets_dw: {targets_dw}")

        targets_dh = wh * torch.log(gt_heights / ex_heights)
        logger.debug(f"\t\t\ttargets_dh = wh * torch.log(gt_heights / ex_heights")
        logger.debug(f"\t\t\t// targets_dh: {targets_dh}\n")

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        logger.debug(f"\t\t\ttargets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)")
        logger.debug(f"\t\t\t// targets: {targets}\n")

        logger.debug(f"\t\t\treturn targets\n")

        logger.debug(f"\t\t}} // END BoxCoder.encode(self, reference_boxes, proposals)\n")
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        logger.debug(f"\t\tBoxCoder.decode(self, rel_codes, boxes) {{ // BEGIN")
        logger.debug(f"\t\t\t// defined in {inspect.getfile(inspect.currentframe())}")
        logger.debug(f"\t\t\t// Params:")
        logger.debug(f"\t\t\t\trel_codes.shape: {rel_codes.shape}")
        logger.debug(f"\t\t\t\tboxes.shape: {boxes.shape}\n")

        boxes = boxes.to(rel_codes.dtype)
        logger.debug(f"\t\t\tboxes = boxes.to(rel_codes.dtype)")
        logger.debug(f"\t\t\t// boxes.shape: {boxes.cpu().shape}\n")

        TO_REMOVE = 1  # TODO remove
        logger.debug(f"\t\t\tTO_REMOVE = 1  # TODO remove\n")

        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        logger.debug(f"\t\t\twidths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE")
        logger.debug(f"\t\t\t// widths.shape: {widths.cpu().shape}\n")

        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        logger.debug(f"\t\t\theights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE")
        logger.debug(f"\t\t\t// heights.shape: {heights.cpu().shape}\n")

        ctr_x = boxes[:, 0] + 0.5 * widths
        logger.debug(f"\t\t\tctr_x = boxes[:, 0] + 0.5 * widths")
        logger.debug(f"\t\t\t// ctr_x.shape: {ctr_x.cpu().shape}\n")

        ctr_y = boxes[:, 1] + 0.5 * heights
        logger.debug(f"\t\t\tctr_y = boxes[:, 1] + 0.5 * heights")
        logger.debug(f"\t\t\t// ctr_y.shape: {ctr_y.cpu().shape}\n\n")

        wx, wy, ww, wh = self.weights
        logger.debug(f"\t\t\twx, wy, ww, wh = self.weights")
        logger.debug(f"\t\t\t// wx: {wx}, wy: {wy}, ww: {ww}, wh: {wh}\n")

        dx = rel_codes[:, 0::4] / wx
        logger.debug(f"\t\t\tdx = rel_codes[:, 0::4] / wx")
        logger.debug(f"\t\t\t// dx.shape: {dx.shape}\n")

        dy = rel_codes[:, 1::4] / wy
        logger.debug(f"\t\t\tdy = rel_codes[:, 1::4] / wy")
        logger.debug(f"\t\t\t// dy.shape: {dy.shape}\n")

        dw = rel_codes[:, 2::4] / ww
        logger.debug(f"\t\t\tdw = rel_codes[:, 2::4] / ww")
        logger.debug(f"\t\t\t// dw.shape: {dw.shape}\n")

        dh = rel_codes[:, 3::4] / wh
        logger.debug(f"\t\t\tdh = rel_codes[:, 3::4] / wh")
        logger.debug(f"\t\t\t// dh.shape: {dh.shape}\n\n")

        # Prevent sending too large values into torch.exp()
        logger.debug(f"\t\t\t# Prevent sending too large values into torch.exp()")

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        logger.debug(f"\t\t\tdw = torch.clamp(dw, max=self.bbox_xform_clip)")
        logger.debug(f"\t\t\t// dw.shape: {dw.shape}\n")

        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        logger.debug(f"\t\t\tdh = torch.clamp(dh, max=self.bbox_xform_clip)")
        logger.debug(f"\t\t\t// dh.shape: {dh.shape}\n")

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        logger.debug(f"\t\t\tpred_ctr_x = dx * widths[:, None] + ctr_x[:, None]")
        logger.debug(f"\t\t\t// pred_ctr_x.shape: {pred_ctr_x.shape}\n")

        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        logger.debug(f"\t\t\tpred_ctr_y = dy * heights[:, None] + ctr_y[:, None]")
        logger.debug(f"\t\t\t// pred_ctr_y.shape: {pred_ctr_y.shape}\n")

        pred_w = torch.exp(dw) * widths[:, None]
        logger.debug(f"\t\t\tpred_w = torch.exp(dw) * widths[:, None]")
        logger.debug(f"\t\t\t// pred_w.shape: {pred_w.shape}\n")

        pred_h = torch.exp(dh) * heights[:, None]
        logger.debug(f"\t\t\tpred_h = torch.exp(dh) * heights[:, None]")
        logger.debug(f"\t\t\t// pred_h.shape: {pred_h.shape}\n\n")

        pred_boxes = torch.zeros_like(rel_codes)
        logger.debug(f"\t\t\tpred_boxes = torch.zeros_like(rel_codes)")
        logger.debug(f"\t\t\t// pred_boxes.shape: {pred_boxes.shape}\n\n")

        # x1
        logger.debug(f"\t\t\t# x1")
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        logger.debug(f"\t\t\tpred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w\n")

        # y1
        logger.debug(f"\t\t\t# y1")
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        logger.debug(f"\t\t\tpred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h\n")

        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        logger.debug(f"\t\t\t# x2 (note: '- 1' is correct; don't be fooled by the asymmetry)")
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        logger.debug(f"\t\t\tpred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1\n")

        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        logger.debug(f"\t\t\t# y2 (note: '- 1' is correct; don't be fooled by the asymmetry)")
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
        logger.debug(f"\t\t\tpred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1\n\n")

        logger.debug(f"\t\t\tpred_boxes.shape: {pred_boxes.shape}\n")

        logger.debug(f"\t\t\treturn pred_boxes\n")
        logger.debug(f"\t\t}} // END BoxCoder.decode(self, rel_codes, boxes)\n")
        return pred_boxes
