import cv2
import torch
import collections
import numpy as np
from math import ceil
from tqdm import tqdm
from itertools import chain
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageDraw
from torch.nn import functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms

from maskrcnn_benchmark.utils.label_catalog import LabelCatalog
from maskrcnn_benchmark.utils.converter import Converter 

class OCRDemo(object):
    def __init__(
        self,
        cfg_det,
        cfg_rec,
        weight_det,
        weight_rec,
    ):
        self.cfg_det = cfg_det.clone()
        self.cfg_rec = cfg_rec.clone()

        self.model_det = build_detection_model(self.cfg_det)
        self.model_rec = build_detection_model(self.cfg_rec)
        self.device = torch.device(cfg_det.MODEL.DEVICE)
        self.model_det.to(self.device)
        self.model_rec.to(self.device)

        checkpointer_det = DetectronCheckpointer(cfg_det, self.model_det, save_dir='/dev/null')
        checkpointer_rec = DetectronCheckpointer(cfg_rec, self.model_rec, save_dir='/dev/null')
        _ = checkpointer_det.load(weight_det)
        _ = checkpointer_rec.load(weight_rec)
        self.model_det.eval()
        self.model_rec.eval()

        self.transforms_det = build_transforms(self.cfg_det, is_recognition=False)
        self.transforms_rec = build_transforms(self.cfg_rec, is_recognition=True)
        self.cpu_device = torch.device("cpu")

        characters = LabelCatalog.get(self.cfg_rec.MODEL.TEXT_RECOGNIZER.CHARACTER)
        self.converter = Converter(characters)

        self.length_for_pred = torch.IntTensor([cfg_rec.MODEL.TEXT_RECOGNIZER.BATCH_MAX_LENGTH])
        self.det_score_thresh = self.cfg_det.TEST.SCORE_THRESHOLD
        self.rec_batch_size = self.cfg_rec.TEST.IMS_PER_BATCH
        self.nms_thresh = self.cfg_det.TEST.NMS_THRESH
        self.box_pad = 0.02

        self.font = ImageFont.truetype('etc/fonts/NotoSansCJKkr-Medium.otf', 10)

    def run_on_pil_image(self, image):
        prediction_whole_image = self.compute_prediction(image)
        image_vis = self.visualize_prediction(image, prediction_whole_image)
        return prediction_whole_image, image_vis

    def compute_prediction(self, image):
        pred = self.detect_prediction(image)
        pred = self.mid_process(pred, image)
        pred = self.recognize_prediction(pred, image)
        return pred

    def detect_prediction(self, image):
        image = self.transforms_det(image)
        image_list = to_image_list(image, self.cfg_det.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            pred_det = self.model_det(image_list)
            pred_det = pred_det[0].to(self.cpu_device)

        filter_thres = pred_det.get_field('scores') > self.det_score_thresh
        pred_det = pred_det[filter_thres]
        return pred_det

    def mid_process(self, pred_det, image):
        pred_det = pred_det.resize(image.size)
        pred_det = boxlist_nms(pred_det, self.nms_thresh)
        pred_det.bbox = pred_det.bbox.numpy().astype(np.int32)
        pred_det.extra_fields['scores'] = pred_det.extra_fields['scores'].numpy()
        return pred_det

    def recognize_prediction(self, pred_det, image):
        result_string = dict()
        rec_img = dict()
        word_images = self.batch_crop_bbox_parallel(image, pred_det.bbox)
        rec_img = {i: self.transforms_rec(word_image) \
            for i, word_image in enumerate(word_images)}

        rec_img_keys_list = list(rec_img.keys())
        rec_img_list = list(rec_img.values())
        for i in range(ceil(len(rec_img)/self.rec_batch_size)):
            start = i * self.rec_batch_size
            end = (i + 1) * self.rec_batch_size
            keys = rec_img_keys_list[start:end]
            images = rec_img_list[start:end]

            trans_list = list()
            with torch.no_grad():
                rec_det = self.model_rec([_.to(self.device) for _ in images])
                for rec in rec_det:
                    trans = self.converter.decode(rec.to(self.cpu_device).get_field('pred'))
                    trans = trans[:trans.find('[s]')]
                    trans_list.append(trans)

            for j, key in enumerate(keys):
                result_string[key] = trans_list[j]

        result_string = collections.OrderedDict(sorted(result_string.items()))
        assert len(pred_det) == len(result_string)    
        pred_det.add_field('texts', list(result_string.values()))
        return pred_det

    def batch_crop_bbox_parallel(self, image, bboxes):
        width, height = image.size
        box_width = bboxes[:,2] - bboxes[:,0]
        box_height = bboxes[:,3] - bboxes[:,1]
        w_padding = np.floor(box_width * self.box_pad).astype(np.int64)
        h_padding = np.floor(box_height * self.box_pad).astype(np.int64)

        bboxes[:,0] -= w_padding
        bboxes[:,1] -= h_padding
        bboxes[:,2] += w_padding
        bboxes[:,3] += h_padding

        bboxes[:,0:2] = np.maximum(bboxes[:,0:2], 0)
        bboxes[:,2] = np.minimum(bboxes[:,2], width-1)
        bboxes[:,3] = np.minimum(bboxes[:,3], height-1)

        image_list = [image.crop(box) for box in bboxes]
        return image_list

    def visualize_prediction(self, image, prediction):
        rects = prediction.bbox
        texts = prediction.get_field('texts')
        for rect, text in zip(rects, texts):
            draw = ImageDraw.Draw(image)
            draw.text((rect[0], rect[1] - 15), text, font=self.font, fill=(255,255,255), stroke_fill=(0,255,0))
            draw.rectangle([*rect], outline=(0, 255, 0))
        return image

if __name__ == '__main__':
    import os
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_det', default='model/detection/config.yaml')
    parser.add_argument('--config_rec', default='model/recognition/config.yaml')
    parser.add_argument('--weight_det', default='model/detection/model.pth')
    parser.add_argument('--weight_rec', default='model/recognition/model.pth')
    parser.add_argument('--dir_src', required=True)
    parser.add_argument('--dir_dst', default='')
    args = parser.parse_args()

    cfg_det = cfg.clone()
    cfg_rec = cfg.clone()
    cfg_det.merge_from_file(args.config_det)
    cfg_rec.merge_from_file(args.config_rec)

    os.makedirs(args.dir_dst, exist_ok=True)
    ocr_demo = OCRDemo(cfg_det, cfg_rec, args.weight_det, args.weight_rec)

    imglist = [_ for _ in os.listdir(args.dir_src) if os.path.splitext(_)[1].lower() in ['.jpg', '.png', '.jpeg', '.tif']]
    imglist.sort()
    for imgname in tqdm(imglist):
        print(imgname)
        image = cv2.imread(os.path.join(args.dir_src, imgname))
        image = Image.fromarray(image, 'RGB')
        prediction, image_vis = ocr_demo.run_on_pil_image(image)
        image_vis = np.asarray(image_vis)
        cv2.imwrite(os.path.join(args.dir_dst, imgname), image_vis)
        print(prediction.get_field('texts'))
