# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import inspect

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from .utils import permute_and_flatten

#for model debugging loe
import logging
from model_log import logger

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
        fpn_post_nms_per_batch=True,
    ):

        if logger.level == logging.DEBUG:
            logger.debug(f"RPNPostProcessing.__init__() {{ //BEGIN")
            logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}")

        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

        if logger.level == logging.DEBUG:
            logger.debug(f"}} // END RPNPostProcessing.__init__()")

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        if logger.level == logging.DEBUG:
            logger.debug(f"RPNPostProcessing.add_gt_proposals() {{ // BEGIN")
            logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}")

        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        if logger.level == logging.DEBUG:
            logger.debug(f"}} // END RPNPostProcessing.add_gt_proposals()")
        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        if logger.level == logging.DEBUG:
            logger.debug(f"\n\t\tRPNPostProcessing.forward_for_single_feature_map() {{ \\ BEGIN")
            logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}")

        device = objectness.device
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        objectness = objectness.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]

        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)

        result = []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            boxlist.add_field("objectness", score)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist = boxlist_nms(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)

        if logger.level == logging.DEBUG:
            logger.debug(f"return result")
            logger.debug(f"\n\t\t}} // END RPNPostProcessing.forward_for_single_feature_map()")

        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        if logger.level == logging.DEBUG:
            logger.debug(f"\t\tRPNPostProcessing.forward(self. anchors, objectness, box_regression, targets=None) {{ // BEGIN")
            logger.debug(f"\t\tParams:")
            logger.debug(f"\t\ttanchors: len(anchors) : {len(anchors)}")
            logger.debug(f"\t\ttobjectness: len(objectness) : {len(objectness)}")
            logger.debug(f"\t\ttbox_regression: type(box_regression) : {type(box_regression)}")
            logger.debug(f"\t\tttarget: {targets}")

        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        if logger.level == logging.DEBUG:
            logger.debug(f"return boxlists")
            logger.debug(f"\t\t}} // END RPNPostProcessing.forward(self. anchors, objectness, box_regression, targets=None)")

        return boxlists

    def select_over_all_levels(self, boxlists):

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\tRPNPostProcessing.select_over_all_levels() {{ // BEGIN")

        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # NOTE: it should be per image, and not per batch. However, to be consistent 
        # with Detectron, the default is per batch (see Issue #672)
        if self.training and self.fpn_post_nms_per_batch:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.bool)
            inds_mask[inds_sorted] = True
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]

        if logger.level == logging.DEBUG:
            logger.debug(f"\t\t}} // END RPNPostProcessing.fselect_over_all_levels()")

        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder):

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\tmake_rpn_postprocessor(config, rpn_box_coder) {{ //BEGIN")
        logger.debug(f"\t\t\tParam:")
        logger.debug(f"\t\t\t\trpn_box_coder: {rpn_box_coder}")

    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    fpn_post_nms_per_batch = config.MODEL.RPN.FPN_POST_NMS_PER_BATCH
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        fpn_post_nms_per_batch=fpn_post_nms_per_batch,
    )

    if logger.level == logging.DEBUG:
        logger.debug(f"\t\t\treturn box_selector")
        logger.debug(f"\t\t\tbox_selector: {box_selector}")
        logger.debug(f"\t\t}} // END make_rpn_postprocessor()")

    return box_selector
