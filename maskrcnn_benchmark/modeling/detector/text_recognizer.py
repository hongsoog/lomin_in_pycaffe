import os
import torch
import numpy as np
import logging

from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.sequence import build_transformer_native
from maskrcnn_benchmark.utils.label_catalog import LabelCatalog

class TextRecognizer(nn.Module):
    def __init__(self, cfg):
        super(TextRecognizer, self).__init__()
        characters = LabelCatalog.get(cfg.MODEL.TEXT_RECOGNIZER.CHARACTER)
        transformer_fsize = cfg.MODEL.TEXT_RECOGNIZER.TRANSFORMER_FSIZE
        cnn_num_pooling = cfg.MODEL.TEXT_RECOGNIZER.CNN_NUM_POOLING
        self.batch_max_length = cfg.MODEL.TEXT_RECOGNIZER.BATCH_MAX_LENGTH
        self.use_projection = cfg.MODEL.TEXT_RECOGNIZER.USE_PROJECTION

        self.backbone = build_backbone(cfg)
        self.OutConv = nn.Conv2d(
            self.backbone.out_channels, transformer_fsize, 
            kernel_size=cfg.MODEL.TEXT_RECOGNIZER.OUTCONV_KS, 
            stride=1, padding=0, bias=False)

        if self.use_projection:
            vfea_len = cfg.INPUT.FIXED_SIZE[0]//(2** cnn_num_pooling) \
                + cfg.INPUT.FIXED_SIZE[1]//(2** cnn_num_pooling)
            cnn_output_channel = transformer_fsize
            Projectio_W = [
                nn.Conv2d(cnn_output_channel, cnn_output_channel, 3, padding=1),
                nn.BatchNorm2d(cnn_output_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(cnn_output_channel, cnn_output_channel, 3, padding=1),
                nn.BatchNorm2d(cnn_output_channel),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((None, 1))
            ]
            Projectio_H = [
                nn.Conv2d(cnn_output_channel, cnn_output_channel, 3, padding=1),
                nn.BatchNorm2d(cnn_output_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(cnn_output_channel, cnn_output_channel, 3, padding=1),
                nn.BatchNorm2d(cnn_output_channel),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, None))
            ]
            self.Projectio_W = nn.Sequential(*Projectio_W)
            self.Projectio_H = nn.Sequential(*Projectio_H)
        else:
            vfea_len = cfg.INPUT.FIXED_SIZE[0] * cfg.INPUT.FIXED_SIZE[1] // (2 ** (cnn_num_pooling * 2))

        encoder_num = cfg.MODEL.TEXT_RECOGNIZER.TRANSFORMER_MODULE_NO
        decoder_num = cfg.MODEL.TEXT_RECOGNIZER.TRANSFORMER_MODULE_NO
        if cfg.MODEL.TEXT_RECOGNIZER.TRANSFORMER_ENCODER_MODULE_NO > 0:
            encoder_num = cfg.MODEL.TEXT_RECOGNIZER.TRANSFORMER_ENCODER_MODULE_NO
        if cfg.MODEL.TEXT_RECOGNIZER.TRANSFORMER_DECODER_MODULE_NO > 0:
            decoder_num = cfg.MODEL.TEXT_RECOGNIZER.TRANSFORMER_DECODER_MODULE_NO

        self.no_recurrent = cfg.MODEL.TEXT_RECOGNIZER.TRANSFORMER_NO_RECURRENT_PATH
        self.SequenceModeling = build_transformer_native(num_classes=len(characters) + 3, 
                                                         vfea_len=vfea_len, 
                                                         num_encoder_layers=encoder_num,
                                                         num_decoder_layers=decoder_num,
                                                         dim_feedforward=transformer_fsize,
                                                         max_len=self.batch_max_length,
                                                         no_recurrent = self.no_recurrent)
        
        if self.no_recurrent:
            self.tgt_mask = None
            self.index_encode = torch.arange(
                start=0, end=self.batch_max_length+1, step=1, dtype=torch.long, 
                device=torch.device('cuda'), requires_grad=False) 

    def forward(self, images):
        images = to_image_list(images)
        input_tensor = images.tensors

        visual_feature = self.backbone(input_tensor)[0]
        visual_feature = self.OutConv(visual_feature)
        
        if self.use_projection:
            visual_feature_h = self.Projectio_W(visual_feature).view(visual_feature.shape[0],visual_feature.shape[1],visual_feature.shape[2])   # [b, c, h]
            visual_feature_w = self.Projectio_H(visual_feature).view(visual_feature.shape[0],visual_feature.shape[1],visual_feature.shape[3])   # [b, c, w]
            visual_feature = torch.cat((visual_feature_h, visual_feature_w), 2).permute(2, 0, 1)
        else:
            visual_feature = visual_feature.permute(3, 0, 2, 1)
            visual_feature = visual_feature.contiguous().view(visual_feature.shape[0],visual_feature.shape[1], -1)

        if self.no_recurrent:
            text = self.index_encode.repeat(visual_feature.size(1), 1).permute(1, 0)
            prediction = self.SequenceModeling(visual_feature, text, None, None)
            result = self.SequenceModeling.generator(prediction)
            score, pred = result.max(2)
            score = score.permute(1, 0)
            pred = pred.permute(1, 0)
        else:
            score, pred = self.SequenceModeling.greedy_decode(visual_feature, self.batch_max_length)

        score = torch.exp(score)
        size = tuple(images.tensors.shape[2:])
        result_list = []
        for i in range(pred.shape[0]):
            _result = BoxList(torch.zeros(1,4),size)
            _result.add_field('pred', pred[i])
            _result.add_field('score', score[i])
            _result.add_field('input_image', images.tensors[i])
            result_list.append(_result)

        return result_list
