# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from .utils import permute_and_flatten

#for model debugging log
import logging
import inspect
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

        logger.debug(f"RPNPostProcessor.__init__() {{ //BEGIN")
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
            logger.debug(f"}} // END RPNPostProcessor.__init__()")

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        if logger.level == logging.DEBUG:
            logger.debug(f"RPNPostProcessor.add_gt_proposals() {{ // BEGIN")
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
            logger.debug(f"}} // END RPNPostProcessor.add_gt_proposals()")
        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        logger.debug(f"\n\t\tRPNPostProcessor.forward_for_single_feature_map() {{ \\ BEGIN")
        logger.debug(f"\t\t\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t\t\t// Params:")
        logger.debug(f"\t\t\t\t\t// anchors: {anchors}")
        logger.debug(f"\t\t\t\t\t// objectness.shape: {objectness.shape}")
        logger.debug(f"\t\t\t\t\t// box_regression.shape: {box_regression.shape}")

        device = objectness.device
        logger.debug(f"\t\t\t\tdevice = objectness.device")
        logger.debug(f"\t\t\t\t// device: {device}\n")

        N, A, H, W = objectness.shape
        logger.debug(f"\t\t\t\tN, A, H, W = objectness.shape")
        logger.debug(f"\t\t\t\t// N: {N}, A:{A}, H:{H}, W:{W}\n")

        # put in the same format as anchors
        logger.debug(f"\t\t\t\t# put objectness in the same format as anchors")
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        logger.debug(f"\t\t\t\tobjectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)")
        logger.debug(f"\t\t\t\t// objectness.shape: {objectness.shape})\n")

        objectness = objectness.sigmoid()
        logger.debug(f"\t\t\t\tobjectness = objectness.sigmoid()")
        logger.debug(f"\t\t\t\t// objectness.shape: {objectness.shape})\n")

        logger.debug(f"\t\t\t\t# put box_regression in the same format as anchors")
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        logger.debug(f"\t\t\t\tbox_regression = permute_and_flatten(box_regression, N, A, 4, H, W)")
        logger.debug(f"\t\t\t\t// box_regression.shape: {box_regression.shape}\n")

        num_anchors = A * H * W
        logger.debug(f"\t\t\t\tnum_anchors = A * H * W")
        logger.debug(f"\t\t\t\t// num_anchors: {num_anchors}\n")

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        logger.debug(f"\t\t\t\tpre_nms_top_n = min(self.pre_nms_top_n, num_anchors)")
        logger.debug(f"\t\t\t\tpre_nms_top_n: {pre_nms_top_n}\n")

        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)
        logger.debug(f"\t\t\t\tobjectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)")
        logger.debug(f"\t\t\t\t// objectness: {objectness}")
        logger.debug(f"\t\t\t\t// topk_idx: {topk_idx}\n")


        batch_idx = torch.arange(N, device=device)[:, None]
        logger.debug(f"\t\t\t\tbatch_idx = torch.arange(N, device=device)[:, None]")
        logger.debug(f"\t\t\t\t// batch_idx: {batch_idx}\n")

        box_regression = box_regression[batch_idx, topk_idx]
        logger.debug(f"\t\t\t\tbox_regression = box_regression[batch_idx, topk_idx]")
        logger.debug(f"\t\t\t\t// box_regression: {box_regression}\n")

        image_shapes = [box.size for box in anchors]
        logger.debug(f"\t\t\t\timage_shapes = [box.size for box in anchors]")
        logger.debug(f"\t\t\t\t// image_shapes: {image_shapes}\n")

        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        logger.debug(f"\t\t\t\tconcat_anchors = torch.cat([a.bbox for a in anchors], dim=0)")
        logger.debug(f"\t\t\t\tconcat_anchors: {concat_anchors}\n")

        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]
        logger.debug(f"\t\t\t\tconcat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]")
        logger.debug(f"\t\t\t\tconcat_anchors: {concat_anchors}\n")

        logger.debug(f"\t\t\t\tproposals = self.box_coder.decode( box_regression.view(-1, 4), concat_anchors.view(-1, 4) ) // CALL")
        logger.debug(f"\t\t\t\t{{")
        proposals = self.box_coder.decode( box_regression.view(-1, 4), concat_anchors.view(-1, 4) )
        logger.debug(f"\n")
        logger.debug(f"\t\t\t\t}}")
        logger.debug(f"\t\t\t\tproposals = self.box_coder.decode( box_regression.view(-1, 4), concat_anchors.view(-1, 4) ) // RETURNED\n")
        logger.debug(f"\t\t\t\t// proposals: {proposals}\n")

        proposals = proposals.view(N, -1, 4)
        logger.debug(f"\t\t\t\tproposals = proposals.view(N, -1, 4)")
        logger.debug(f"\t\t\t\t// proposals: {proposals}\n")

        result = []
        logger.debug(f"\t\t\t\tresult = []")

        logger.debug(f"\t\t\t\tfor proposal, score, im_shape in zip(proposals, objectness, image_shapes):\n\t\t\t\t{{")
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            logger.debug(f"\t\t\t\t\t#================================")
            logger.debug(f"\t\t\t\t\t# proposals: {proposal}")
            logger.debug(f"\t\t\t\t\t# score: {score}")
            logger.debug(f"\t\t\t\t\t# im_shape: {im_shape}")
            logger.debug(f"\t\t\t\t\t#================================")

            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            logger.debug(f'\t\t\t\t\tboxlist = BoxList(proposal, im_shape, mode="xyxy")')

            boxlist.add_field("objectness", score)
            logger.debug(f'\t\t\t\t\tboxlist.add_field("objectness", score)')

            boxlist = boxlist.clip_to_image(remove_empty=False)
            logger.debug(f"\t\t\t\t\tboxlist = boxlist.clip_to_image(remove_empty=False)")

            boxlist = remove_small_boxes(boxlist, self.min_size)
            logger.debug(f"\t\t\t\t\tboxlist = remove_small_boxes(boxlist, self.min_size)")

            logger.debug(f"\t\t\t\t\tboxlist_nms() // CALL\n\t\t\t\t\t{{")
            boxlist = boxlist_nms( boxlist, self.nms_thresh, max_proposals=self.post_nms_top_n, score_field="objectness", )
            logger.debug(f"\n\t\t\t\t\t}} boxlist_nms() // RETURNED\n")

            result.append(boxlist)
            logger.debug(f"\t\t\t\t\tresult.append(boxlist)")

        logger.debug(f"\n\t\t\t\t}} // END for proposal, score, im_shape in zip(proposals, objectness, image_shapes)\n")

        logger.debug(f"return result")
        logger.debug(f"\n\t\t}} // END RPNPostProcessor.forward_for_single_feature_map()")

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
        logger.debug(f"\t\tRPNPostProcessor.forward(self. anchors, objectness, box_regression, targets=None) {{ // BEGIN")
        logger.debug(f"\t\t\t\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t\t\t\t// Params:")
        logger.debug(f"\t\t\t\t\t// len(anchors): {len(anchors)}")
        logger.debug(f"\t\t\t\t\t// anchors: {anchors}")
        logger.debug(f"\n\t\t\t\t\t// len(objectness): {len(objectness)} aka box_cls")
        for i, obj in enumerate(objectness):
            logger.debug(f"\t\t\t\t\t//objectness[{i}].shape): {obj.shape}")

        logger.debug(f"\n\t\t\t\t\t// len(box_regression): : {len(box_regression)}")
        for i, box_regr in enumerate(box_regression):
            logger.debug(f"\t\t\t\t\t//box_regression[{i}].shape): {box_regr.shape}")

        logger.debug(f"\t\t\t\t\t// target: {targets}\n")

        sampled_boxes = []
        logger.debug(f"\t\t\t\tsampled_boxes = []")

        num_levels = len(objectness)
        logger.debug(f"\t\t\t\tnum_levels = len(objectness)")
        logger.debug(f"\t\t\t\t// num_levels: {num_levels}\n")

        anchors = list(zip(*anchors))
        logger.debug(f"\t\t\t\tanchors = list(zip(*anchors))\n")

        logger.debug(f"\t\t\t\tfor a, o, b in zip(anchors, objectness, box_regression) {{" )
        i = 1
        total_num_iter = len(anchors)

        logger.debug(f"\t\t\t\t\t# 2-3-2-3-1 loop over single_feature_map")
        for a, o, b in zip(anchors, objectness, box_regression):
            logger.debug(f"\t\t\t\t\t{{")
            logger.debug(f"\t\t\t\t\t# BEGIN for a, o, b in zip()  iteration: {i}/{total_num_iter}")
            logger.debug(f"\t\t\t\t\t# a: {a}")
            logger.debug(f"\t\t\t\t\t# o.shape: {o.shape}")
            logger.debug(f"\t\t\t\t\t# b.shape: {b.shape}\n")

            logger.debug(f"\t\t\t\t\t# 2-3-2-3-2 self.forward_for_single_feature_map")
            logger.debug(f"\t\t\t\t\t// self.forward_for_single_feature_map: {self.forward_for_single_feature_map}")
            logger.debug(f"\t\t\t\t\tsampled_boxes.append(self.forward_for_single_feature_map(a, o, b)) {{ // CALL")

            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

            logger.debug(f"\t\t\t\t\t}}")
            logger.debug(f"\t\t\t\t\tsampled_boxes.append(self.forward_for_single_feature_map(a, o, b)) // RETURNED\n")

            logger.debug(f"\t\t\t\t\t}} // END of for a, o, b in zip() iteration: {i}/{total_num_iter}\n")
            i = i + 1

        logger.debug(f"\t\t\t\t}}// END for a, o, b in zip(anchors, objectness, box_regression)\n" )

        logger.debug(f"\t\t\t\t# 2-3-2-3-2 boxlists = list(zip(*sampled_boxes))")
        boxlists = list(zip(*sampled_boxes))
        logger.debug(f"\t\t\t\tboxlists = list(zip(*sampled_boxes))")
        logger.debug(f"\t\t\t\tboxlists: {boxlists}\n")

        logger.debug(f"\t\t\t\t# 2-3-2-3-3 boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]\n")
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        logger.debug(f"\t\t\t\tboxlists = [cat_boxlist(boxlist) for boxlist in boxlists]\n")
        logger.debug(f"\t\t\t\tboxlists: {boxlists}\n")


        if num_levels > 1:
            logger.debug(f"\t\t\t\tif num_levels > 1:")
            logger.debug(f"\t\t\t\t\t# 2-3-2-3-4 boxlists = self.select_over_all_levels(boxlists)")
            logger.debug(f"\t\t\t\t\tboxlists = self.select_over_all_levels(boxlists) {{ // CALL")
            boxlists = self.select_over_all_levels(boxlists)
            logger.debug(f"\t\t\t\t\t}}")
            logger.debug(f"\t\t\t\t\tboxlists = self.select_over_all_levels(boxlists) // RETURNED")

        # append ground-truth bboxes to proposals
        logger.debug(f"\t\t\t\tif self.training: {self.training} and targets: {targets} is not None:")
        if self.training and targets is not None:
            logger.debug(f"\t\t\t\t\tboxlists = self.add_gt_proposals(boxlists, targets) {{ // CALL")
            boxlists = self.add_gt_proposals(boxlists, targets)
            logger.debug(f"\t\t\t\t\t}}")
            logger.debug(f"\t\t\t\t\tboxlists = self.add_gt_proposals(boxlists, targets) // RETURNED")

        logger.debug(f"\t\t\t\t// boxlists: {boxlists}\n")

        logger.debug(f"\t\t\t\treturn boxlists\n")
        logger.debug(f"\t\t}} // END RPNProcessor.forward(self. anchors, objectness, box_regression, targets=None)")

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
