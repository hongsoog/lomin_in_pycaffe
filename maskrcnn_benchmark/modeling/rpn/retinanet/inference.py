import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

#for model debugging log
import logging
import inspect
from model_log import logger

class RetinaNetPostProcessor(RPNPostProcessor):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder,
        num_classes,
        fpn_post_nms_top_n,
    ):
        super(RetinaNetPostProcessor, self).__init__(
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=nms_thresh,
            min_size=min_size,
            box_coder=box_coder,
            fpn_post_nms_top_n=fpn_post_nms_top_n,
        )

        logger.debug(f"RetinaNetPostProcessor.__init__() {{ //BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t// Params:")
        logger.debug(f"\t\t> pre_nms_top_n: {pre_nms_top_n}")
        logger.debug(f"\t\t> post_nms_top_n: {post_nms_top_n}")
        logger.debug(f"\t\t> nms_thresh: {nms_thresh}")
        logger.debug(f"\t\t> min_size: {min_size}")
        logger.debug(f"\t\t> box_coder: {box_coder}")
        logger.debug(f"\t\t> fpn_post_nms_top_n: {fpn_post_nms_top_n}")

        self.pre_nms_thresh = pre_nms_thresh
        logger.debug(f"\tself.pre_nms_thresh = pre_nms_thresh")

        self.pre_nms_top_n = pre_nms_top_n
        logger.debug(f"\tself.pre_nms_top_n = pre_nms_top_n")

        self.nms_thresh = nms_thresh
        logger.debug(f"\tself.nms_thresh = nms_thresh")

        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        logger.debug(f"\tself.fpn_post_nms_top_n = fpn_post_nms_top_n")

        self.min_size = min_size
        logger.debug(f"\tself.min_size = min_size")

        self.num_classes = num_classes
        logger.debug(f"\tself.num_classes = num_classes")

        if box_coder is None:
            logger.debug(f"\tif box_coder is None:")
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
            logger.debug(f"\t\tbox_coder = BoxCoder(weights=(10., 10., 5., 5.))")

        self.box_coder = box_coder
        logger.debug(f"\tself.box_coder = box_coder\n")
        logger.debug(f"}}// END RetinaNetPostProcessor.__init__()")

    def add_gt_proposals(self, proposals, targets):
        """
        This function is not used in RetinaNet
        """
        pass

    def forward_for_single_feature_map(
            self, anchors, box_cls, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        logger.debug(f"RetinaNetPostProcessor.forward_for_single_feature_map() {{ //BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t// Params:")
        logger.debug(f"\t\t> anchors: list[BoxList]")
        logger.debug(f"\t\t> box_cls: tensor of size N, A*C, H, W)")
        logger.debug(f"\t\t> box_regression: tensor of size N, A*4, H, W)\n")

        device = box_cls.device
        logger.debug(f"\tdevice = box_cls.device")
        logger.debug(f"\t// device: {device}\n")

        N, _, H, W = box_cls.shape
        logger.debug(f"\tN, _, H, W = box_cls.shape")
        logger.debug(f"\t// N:{N}, H:{H}, W:{W}\n")

        A = box_regression.size(1) // 4
        logger.debug(f"\tA = box_regression.size(1) // 4")
        logger.debug(f"\t// A : {A}\n")

        C = box_cls.size(1) // A
        logger.debug(f"\tC = box_cls.size(1) // A")
        logger.debug(f"\t// C: {C}\n")

        # put box_cls in the same format as anchors
        logger.debug(f"\t# put 'box_cls' in the same format as anchors")

        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        logger.debug(f"\tbox_cls = permute_and_flatten(box_cls, N, A, C, H, W)")
        logger.debug(f"\t// box_cls.shape: {box_cls.shape}\n")

        box_cls = box_cls.sigmoid()
        logger.debug(f"\tbox_cls = box_cls.sigmoid()")
        logger.debug(f"\t// box_cls.shape: {box_cls.shape}\n")


        logger.debug(f"\t# put 'box_cls' in the same format as anchors")

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        logger.debug(f"\tbox_regression = permute_and_flatten(box_regression, N, A, 4, H, W)")
        logger.debug(f"\t// box_regression.shape: {box_regression.shape}\n")

        box_regression = box_regression.reshape(N, -1, 4)
        logger.debug(f"\tbox_regression = box_regression.reshape(N, -1, 4)")
        logger.debug(f"\t// box_regression.shape: {box_regression.shape}\n")

        num_anchors = A * H * W
        logger.debug(f"\tnum_anchors = A * H * W")
        logger.debug(f"\t// num_anchors: {num_anchors}\n")

        candidate_inds = box_cls > self.pre_nms_thresh
        logger.debug(f"\tcandidate_inds = box_cls > self.pre_nms_thresh")
        logger.debug(f"\t// candidate_inds.shape: {candidate_inds.shape}\n")

        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        logger.debug(f"\tpre_nms_top_n = candidate_inds.view(N, -1).sum(1)")
        logger.debug(f"\t// pre_nms_top_n: {pre_nms_top_n}\n")

        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        logger.debug(f"\tpre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)")
        logger.debug(f"\t// pre_nms_top_n: {pre_nms_top_n}\n")

        results = []
        logger.debug(f"\tresults = []\n")

        logger.debug(f"\t// box_cls.shape: {box_cls.shape}")
        logger.debug(f"\t// box_regression.shape: {box_regression.shape}")
        logger.debug(f"\t// pre_nmns_top_n: {pre_nms_top_n}")
        logger.debug(f"\t// candidate.inds.shape: {candidate_inds.shape}")
        logger.debug(f"\t// anchors: {anchors}")
        logger.debug(f"\tfor per_box_cls, per_box_regression, ... in zip():\n\t{{")
        for per_box_cls, per_box_regression, per_pre_nms_top_n, \
        per_candidate_inds, per_anchors in zip(
            box_cls,
            box_regression,
            pre_nms_top_n,
            candidate_inds,
            anchors):

            # Sort and select TopN
            # TODO most of this can be made out of the loop for all images.
            # TODO:Yang: Not easy to do. Because the numbers of detections are
            # different in each image. Therefore, this part needs to be done
            # per image.

            logger.debug(f"\t\t# ====================================")
            logger.debug(f"\t\t# per_box_cls.shape: {per_box_cls.shape}")
            logger.debug(f"\t\t# type(per_box_cls): {type(per_box_cls)}")
            logger.debug(f"\t\t# per_box_regression.shape: {per_box_regression.shape}")
            logger.debug(f"\t\t# per_pre_nms_top_n: {per_pre_nms_top_n}")
            logger.debug(f"\t\t# per_candidate_inds.shape: {per_candidate_inds.shape}")
            logger.debug(f"\t\t# per_anchors: {per_anchors}")
            logger.debug(f"\t\t# ====================================\n")

            per_box_cls = per_box_cls[per_candidate_inds]
            logger.debug(f"\t\tper_box_cls = per_box_cls[per_candidate_inds]")

            # see Torch.topk() method
            # topk(k, sorted=False)
            #   returns namedTuples of (value, indices), where the indices are the indices of
            #   the elements in the original input tensor.
            #   boolean option sorted if True, will make sure that the returned k elements are tehmselves sorted.
            per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
            logger.debug(f"\t\tper_box_cls, top_k_indices =per_box_cls.topk(per_pre_nms_top_n, sorted=False)")
            logger.debug(f"\t\t// per_box_cls: {per_box_cls.cpu()}")
            logger.debug(f"\t\t// top_k_indices: {top_k_indices.cpu()}\n")


            per_candidate_nonzeros = \
                    per_candidate_inds.nonzero()[top_k_indices, :]

            logger.debug(f"\t\tper_candidate_nonzeros = \\")
            logger.debug(f"\t\t   per_candidate_inds.nonzero()[top_k_indices, :]")
            logger.debug(f"\t\t// per_candidate_inds.shape: {per_candidate_inds.shape}\n")

            per_box_loc = per_candidate_nonzeros[:, 0]
            logger.debug(f"\t\tper_box_loc = per_candidate_nonzeros[:, 0]")
            logger.debug(f"\t\t// per_box_loc.shape: {per_box_loc.shape}\n")

            per_class = per_candidate_nonzeros[:, 1]
            logger.debug(f"\t\tper_class = per_candidate_nonzeros[:, 1]")
            logger.debug(f"\t\t// per_class.shape: {per_class.shape}\n")

            per_class += 1
            logger.debug(f"\t\tper_class += 1\n")

            logger.debug(f"\t\tdetections = self.box_coder.decode( ) {{ // CALL")
            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )
            logger.debug(f"\t\t}} detections = self.box_coder.decode( ) // RETURNED\n")
            logger.debug(f"\t\t// type(detections): {type(detections)}")
            logger.debug(f"\t\t// detections: {detections.cpu()}\n")

            logger.debug(f'\t\tboxlist = BoxList(detections, per_anchors.size, mode="xyxy") {{ // CALL')

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")

            logger.debug(f'\t\t}} boxlist = BoxList(detections, per_anchors.size, mode="xyxy") // RETURNED\n')
            logger.debug(f'\t\t// type(boxlist): {type(boxlist)}')
            logger.debug(f'\t\t// boxlist: {boxlist}\n')

            boxlist.add_field("labels", per_class)
            logger.debug(f'\t\tboxlist.add_field("labels", per_class)')

            boxlist.add_field("scores", per_box_cls)
            logger.debug(f'\t\tboxlist.add_field("scores", per_box_cls)')

            boxlist = boxlist.clip_to_image(remove_empty=False)
            logger.debug(f"\t\tboxlist = boxlist.clip_to_image(remove_empty=False)")

            boxlist = remove_small_boxes(boxlist, self.min_size)
            logger.debug(f"\t\tboxlist = remove_small_boxes(boxlist, self.min_size)")

            results.append(boxlist)
            logger.debug(f"\t\tresults.append(boxlist)")

        logger.debug(f"\t}} // END for per_box_cls, per_box_regression, ... in zip():\n")

        logger.debug(f"\treturn results\n")
        logger.debug(f"}} // END RetinaNetPostProcessor.forward_for_single_feature_map()")
        return results

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        logger.debug(f"RetinaNetPostProcessor.select_over_all_levels() {{ //BEGIN")
        logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
        logger.debug(f"\t// Params:")
        logger.debug(f"\t\t> boxlists:")

        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)

        logger.debug(f"}} // END RetinaNetPostProcessor.select_over_all_levels()")
        return results


def make_retinanet_postprocessor(config, rpn_box_coder):
    logger.debug(f"make_retinanet_postprocessor() {{ //BEGIN")
    logger.debug(f"\t// defined in {inspect.getfile(inspect.currentframe())}\n")
    logger.debug(f"\t// Params:")
    logger.debug(f"\t\t> config:")
    logger.debug(f"\t\t> rpn_box_coder:\n")

    pre_nms_thresh = config.MODEL.RETINANET.INFERENCE_TH
    pre_nms_top_n = config.MODEL.RETINANET.PRE_NMS_TOP_N
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RETINANET.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    num_classes = config.MODEL.RETINANET.NUM_CLASSES
    min_size = 0

    box_selector = RetinaNetPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        num_classes=num_classes,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
    )
    logger.debug(f"}} // END make_retinanet_postprocessor()")
    return box_selector