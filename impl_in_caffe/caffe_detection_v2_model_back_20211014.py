import torch
import os
from PIL import Image, ImageDraw
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

import ./caffe_util as cu

prototxt_file = './detection_v2.prototxt'


#--------------------------------------------------
# RsetNet50 (model.backbone) related function
#--------------------------------------------------
# - conv_bn_scale() : conv => bn => scale block creation
# - resnet_stage_sbulayer() : Resnet stage sublayer block creation
#--------------------------------------------------
def conv_bn_scale(bottom, nout, bias_term=False, **kwargs):
    ''' Build a single [Conv -> BN -> Scale] block

        Params:
        bottom (blob): input to this conv/bn/scale block
        nout (uint): num of oputs in conv layer
        bias_term (bool, default:False): use or not bias term in conv layer

        Returns
        top of conv, bn, scale : blob
    '''
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term, **kwargs)

    # https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/batch_norm.py
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    return conv, bn, scale


def resnet_stage_sublayer(layer_name, sublayer_num, n, bottom, nout, downsample_branch=False, initial_stride=2):
    '''Builds a single Resnet Stage
       resetne50 (model.backbone) has 5 stages
       - stage 0 (stem), stage 1, stage 2, stage 3 and stage 4

    Prameters:
       layer_num (uint): layer number as ? in "backbone_body_layer?"
       sub_num (uint): sublayer number as # in "backbone_body_layer?_#"
       n (caffe.NetSpec): NetSpec object
       bottom (Blob): input to this stage sublayer
       nout (uint): num. of out ch.
       downsample (bool) :  inclusion of subsample at beginning of this stage sublayer?
                     if BottleNeck architecutre used, set True (default: False)
       initial_stride (uint, default=2): stride used in branch2a and branc2b. 

       Note that downsample(branch3b) stride is always 1.


    Overview diagram:

             +------> [ *_downsample_0 -- *_downsample_1 --- *_downsample_scale ] -----------+
             |                                                                               |
    bottom --+---------> [ *_conv1 --- *_bn1 --- *_scale1 --- *_relu1 ] ----+                |
             |                                                              |                |
             |     +--------------------------------------------------------+                |
             |     |                                                                         |
             |     +---> [ *_conv2 --- *_bn2 --- *_scale2 --- *_relu2 ] ----+                |
             |                                                              |                | if downsample == True
             |     +--------------------------------------------------------+                |
             |     |                                                                         V
             |     +---> [ *_conv3 --- *_bn3 --- *_scale3 -------------------------> [ *_res: Eltwise ] ----> *__res_relu :ReLu
             |                                                                                ^
             |                                                                                |
             |                                                                                | if downsample == False
             +--------------------------------------------------------------------------------+
    '''
    prefix = f'{layer_name}_{sublayer_num}'

    # -----------------------------------------------------------------
    # input downsampling layer at the begining of every layer[0-4]
    # ------------------------------------------------------------------
    # In pytorch model,
    # backbone.body.laye{layer_num}.{sublayer_num}.downsample_0
    # backbone.body.layer{layer_num}.{sublayer_num}.downsample_1
    # ------------------------------------------------------------------
    if downsample_branch:
        # downsample at first layer in resnet block
        downsample_conv = f'{prefix}_downsample_0'
        downsample_bn = f'{prefix}_downsample_1'
        downsample_scale = f'{prefix}_downsample_scale'
        n[downsample_conv], n[downsample_bn], n[downsample_scale] = \
            conv_bn_scale( bottom, 4*nout, kernel_size=1, stride=initial_stride, pad=0)
    else:
        initial_stride = 1

    # ------------------------------------------------------------------
    # In pytorch model,
    # backbone.body.layer{layer_num}.{sublayer_num}.conv1
    # backbone.body.layer{layer_num}.{sublayer_num}.bn1
    # backbone.body.layer{layer_num}.{sublayer_num}.scale1
    # backbone.body.layer{layer_num}.{sublayer_num}.relu1
    # ------------------------------------------------------------------
    conv1 = f'{prefix}_conv1'
    bn1 = f'{prefix}_bn1'
    scale1 = f'{prefix}_scale1'

    n[conv1], n[bn1], n[scale1] = \
        conv_bn_scale( bottom, nout, kernel_size=1, stride=initial_stride, pad=0)

    relu1 = f'{prefix}_relu1'
    n[relu1] = L.ReLU(n[scale1], in_place=True)

    # ------------------------------------------------------------------
    # In pytorch model,
    # backbone.body.layer{layer_num}.{sublayer_num}.conv2
    # backbone.body.layer{layer_num}.{sublayer_num}.bn2
    # backbone.body.layer{layer_num}.{sublayer_num}.scale2
    # backbone.body.layer{layer_num}.{sublayer_num}.relu2
    # ------------------------------------------------------------------
    conv2 = f'{prefix}_conv2'
    bn2 = f'{prefix}_bn2'
    scale2 = f'{prefix}_scale2'

    n[conv2], n[bn2], n[scale2] = \
        conv_bn_scale( n[relu1], nout, kernel_size=3, stride=1, pad=1)

    relu2 = f'{prefix}_relu2'
    n[relu2] = L.ReLU(n[scale2], in_place=True)

    # ------------------------------------------------------------------
    # In pytorch model,
    # backbone.body.layer{layer_num}.{sublayer_num}.conv3
    # backbone.body.layer{layer_num}.{sublayer_num}.bn3
    # backbone.body.layer{layer_num}.{sublayer_num}.scale3
    # ------------------------------------------------------------------
    conv3 = f'{prefix}_conv3'
    bn3 = f'{prefix}_bn3'
    scale3 = f'{prefix}_scale3'

    n[conv3], n[bn3], n[scale3] = \
        conv_bn_scale( n[relu2], 4 * nout, kernel_size=1, stride=1, pad=0)

    # ---------------------------------------
    # skip connection processing
    # ---------------------------------------
    eltwise = f'{prefix}_eltwise'

    if downsample_branch:
        n[eltwise] = L.Eltwise(n[downsample_scale], n[scale3])
    else:
        n[eltwise] = L.Eltwise(bottom, n[scale3])

    # ---------------------------------------
    # last relu
    # backbone.body.layer{layer_num}.{sublayer_num}.relu
    # ---------------------------------------
    relu = f'{prefix}_relu'
    n[relu] = L.ReLU(n[eltwise], in_place=True)


#--------------------------------------------------
# RPN (model.rpn) related function
#--------------------------------------------------
# - cls_tower_logits() : cls_tower + cls_logits build
# - bbox_tower_pred(): bbox_tower _ bbox_pre build
#--------------------------------------------------
def cls_tower_logits(bottom, nout=9, bias_term=True, **kwargs):
    '''Builds cls_tower_logits in form of [conv => relu]x4 + conv block.
       all conv layer uses kernel:3, stride: 1, pad: 1

       Params:
          bottom (blob) : input to cls tower logits
          nout (uint) : num. of outputs in cls logit (last conv layer)
          bias_trerm (bool) : use bias term or not in conv layer
    '''

    # cls tower: 4 times repetition of [conv => relu]
    conv1 = L.Convolution(bottom, num_output=1024, bias_term=bias_term, **kwargs)
    relu1 = L.ReLU(conv1, in_place=True)

    conv2 = L.Convolution(relu1, num_output=1024, bias_term=bias_term, **kwargs)
    relu2 = L.ReLU(conv2, in_place=True)

    conv3 = L.Convolution(relu2, num_output=1024, bias_term=bias_term, **kwargs)
    relu3 = L.ReLU(conv3, in_place=True)

    conv4 = L.Convolution(relu3, num_output=1024, bias_term=bias_term, **kwargs)
    relu4 = L.ReLU(conv4, in_place=True)

    # cl_logits with nout=9
    cls_logits = L.Convolution(relu4, num_output=nout, **kwargs, bias_term=bias_term)

    return conv1, relu1, conv2, relu2, conv3, relu3, conv4, relu4, cls_logits


def bbox_tower_pred(bottom, nout=36, bias_term=True, **kwargs):
    '''Builds build a bbox_tower_pred in form of [conv => relu]x4 + conv block.
       all conv layer uses kernel:3, stride: 1, pad: 1

       Params:
          bottom (blob) : input to cls tower logits
          nout (uint) : num. of outputs in cls logit (last conv layer)
          bias_term (bool) : use bias term or not in conv layer
    '''
    # bbox_tower: 4 times repetition of [conv => relut]
    conv1 = L.Convolution(bottom, num_output=1024, bias_term=bias_term, **kwargs)
    relu1 = L.ReLU(conv1, in_place=True)

    conv2 = L.Convolution(relu1, num_output=1024, bias_term=bias_term, **kwargs)
    relu2 = L.ReLU(conv2, in_place=True)

    conv3 = L.Convolution(relu2, num_output=1024, bias_term=bias_term, **kwargs)
    relu3 = L.ReLU(conv3, in_place=True)

    conv4 = L.Convolution(relu3, num_output=1024, bias_term=bias_term, **kwargs)
    relu4 = L.ReLU(conv4, in_place=True)

    # bbox_pred with nout=36
    bbox_pred = L.Convolution(relu4, num_output=nout, bias_term=bias_term, **kwargs)

    return conv1, relu1, conv2, relu2, conv3, relu3, conv4, relu4, bbox_pred


#--------------------------------------------------
# detection_network v2 model structure
# model := backbone + rpn (region proposal network)
# backbone := body + fpn (feature pyramid network)
# body := resnet 50
#--------------------------------------------------
def detection_network(n, bottom):
    '''Build Backbone with ResNet50 + FPN

    Params:
    n (caffe.NetSpec) : NetSpec instance
    bottom (blob) : innput to detection network

    '''
    # keep list of feature maps in stage ofrder
    features = []


    # ************************************************
    # 1. model.backbone
    # ************************************************
    # backbone := body + fpn
    # ************************************************


    # =================================================================
    # 1.1 model.backbone.body
    # body = stem (layer 0) +  layer 1 + layer 2 + layer 3 + layer 4
    # =================================================================

    prefix = "backbone_body"
    # -------------------------------------
    # 1.1.0 model.backbone.body.stem (layer0)
    # sublayer: conv1,  bn1, scale, relu, maxpool
    # -------------------------------------
    layer = f'{prefix}_stem_'

    n[layer + 'conv1'], n[layer + 'bn1'], n[layer + 'scale'] = conv_bn_scale(bottom, 64, bias_term=False,
                                                                              kernel_size=7, pad=3, stride=2)
    n[layer + 'relu'] = L.ReLU(n[layer + 'scale'], in_place=True)
    n[layer + 'maxpool'] = L.Pooling(n[layer + 'relu'], kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # -------------------------------------
    # 1.1.1 model.backbone.body.layer1
    # sublayer: 0, 1, 2
    # -------------------------------------
    pre_layer = layer
    layer = f'{prefix}_layer1'

    resnet_stage_sublayer(layer, '0', n, n[pre_layer + 'maxpool'], 64, downsample_branch=True, initial_stride=1)

    #resnet_stage_sublayer(layer_name, sublayer_num, n, bottom, nout, downsample_branch=False, initial_stride=2):
    resnet_stage_sublayer(layer, '1', n, n[layer + '_0_relu'], 64)
    resnet_stage_sublayer(layer, '2', n, n[layer + '_1_relu'], 64)

    # feature C1
    features.append(n[layer + '_2_relu'])

    # -------------------------------------
    # 1.1.2 model.backbone.body.layer2 (stage 2)
    # sublayer: 0, 1, 2, 3
    # -------------------------------------
    pre_layer = layer
    layer = f'{prefix}_layer2'

    resnet_stage_sublayer(layer, '0', n, n[pre_layer + '_2_relu'], 128, downsample_branch=True)
    resnet_stage_sublayer(layer, '1', n, n[layer + '_0_relu'], 128)
    resnet_stage_sublayer(layer, '2', n, n[layer + '_1_relu'], 128)
    resnet_stage_sublayer(layer, '3', n, n[layer + '_2_relu'], 128)

    # feature C2
    features.append(n[layer + '_3_relu'])

    # -------------------------------------
    # 1.1.3 model.backbone.body.layer3 (stage 3)
    # sublayer: 0, 1, 2, 3, 4, 5
    # -------------------------------------
    pre_layer = layer
    layer = f'{prefix}_layer3'

    resnet_stage_sublayer(layer, '0', n, n[pre_layer + '_3_relu'], 256, downsample_branch=True)
    resnet_stage_sublayer(layer, '1', n, n[layer + '_0_relu'], 256)
    resnet_stage_sublayer(layer, '2', n, n[layer + '_1_relu'], 256)
    resnet_stage_sublayer(layer, '3', n, n[layer + '_2_relu'], 256)
    resnet_stage_sublayer(layer, '4', n, n[layer + '_3_relu'], 256)
    resnet_stage_sublayer(layer, '5', n, n[layer + '_4_relu'], 256)

    # feature C3
    features.append(n[layer + '_5_relu'])

    # -------------------------------------
    # 1.1.4 model.backbone.body.layer4 (stage 4)
    # sublayer: 0, 1, 2
    # -------------------------------------
    pre_layer = layer
    layer = f'{prefix}_layer4'

    resnet_stage_sublayer(layer, '0', n, n[pre_layer + '_5_relu'], 512, downsample_branch=True)
    resnet_stage_sublayer(layer, '1', n, n[layer + '_0_relu'], 512)
    resnet_stage_sublayer(layer, '2', n, n[layer + '_1_relu'], 512)

    # feature C4
    features.append(n[layer + '_2_relu'])

    C1, C2, C3, C4 = features

    # =================================================================
    # 1.2 model.backbone.fpn
    # =================================================================
    fpn_results = []
    prefix = "backbone_fpn"

    # -------------------------------------
    # 1.2.1 model.backbone.fpn_inner4
    # -------------------------------------
    layer = f'{prefix}_inner4'
    last_inner = n[layer] = L.Convolution(C4, num_output=1024, kernel_size=1, stride=1)

    # -------------------------------------
    # 1.2.2 model.backbone.fpn_layer4
    # -------------------------------------
    layer = f'{prefix}_layer4'
    n[layer] = L.Convolution(last_inner, num_output=1024, kernel_size=3, stride=1, pad=1)

    # featuer pyramid feature P4 append
    # P4 = conv (conv (C4))
    fpn_results.append(n[layer])

    # -------------------------------------
    # 1.2.3 model.backbone.fpn_inner3_upsample
    # -------------------------------------
    layer = f'{prefix}_inner3_upsample' # inner3__upsample
    inner3_upsample = n[layer] = L.Deconvolution(last_inner,
                               convolution_param
                               = dict(num_output=1024, kernel_size=4, stride=2, pad=1,
                                      weight_filler=dict(type ='bilinear'),
                                      bias_term=False))

    # -------------------------------------
    # 1.2.4 model.backbone.fpn_inner3_lateral
    # -------------------------------------
    layer = f'{prefix}_inner3_lateral'
    inner3_lateral = n[layer] = L.Convolution(C3, num_output=1024, kernel_size=1, stride=1)

    # -------------------------------------
    # 1.2.5 model.backbone.fpn_inner3_lateral_sum
    # -------------------------------------
    layer = f'{prefix}_inner3_lateral_sum' # P3
    last_inner = n[layer] = L.Eltwise(inner3_lateral,  inner3_upsample)

    # -------------------------------------
    # 1.2.6 model.backbone.fpn_layer3
    # -------------------------------------
    layer = f'{prefix}_layer3'
    n[layer] = L.Convolution(last_inner, num_output=1024, kernel_size=3, stride=1, pad=1)

    # feature pyramid P3 insert at idx 0
    # P3 = unsample(P4) + conv(conv(C3))
    fpn_results.insert(0, n[layer])

    # -------------------------------------
    # 1.2.7 model.backbone.fpn_inner2_upsample
    # -------------------------------------
    layer = f'{prefix}_inner2_upsample' # inner2__upsample
    inner2_upsample = n[layer] = L.Deconvolution(last_inner,
                               convolution_param
                               = dict(num_output=1024, kernel_size=4, stride=2, pad=1,
                                      weight_filler=dict(type ='bilinear'),
                                      bias_term=False))

    # -------------------------------------
    # 1.2.8 model.backbone.fpn_inner3_lateral
    # -------------------------------------
    layer = f'{prefix}_inner2_lateral'
    inner2_lateral = n[layer] = L.Convolution(C2, num_output=1024, kernel_size=1, stride=1)

    # -------------------------------------
    # 1.2.9 model.backbone.fpn_inner2_lateral_sum
    # -------------------------------------
    layer = f'{prefix}_inner2_lateral_sum' # P3
    last_inner = n[layer] = L.Eltwise(inner2_lateral,  inner2_upsample)

    # -------------------------------------
    # 1.2.10 model.backbone.fpn_layer2
    # -------------------------------------
    layer = f'{prefix}_layer2'
    n[layer] = L.Convolution(last_inner, num_output=1024, kernel_size=3, stride=1, pad=1)

    # feature pyramid P2 insert at idx 0
    # P2 = unsample(P3) + conv(conv(C2))
    fpn_results.insert(0, n[layer])


    # -------------------------------------
    # 1.2.11 model.fpn.top_blocks
    # -------------------------------------

    # -------------------------------------
    # 1.2.11.1 model.fpn.top_blocks.p6
    # -------------------------------------
    prefix = "backbone_fpn_topblocks"
    layer = f'{prefix}_p6'
    n[layer] = L.Convolution(C4, num_output=1024, kernel_size=3, stride=2, pad=1)

    # feature pyramid P6 append at end
    # P6 = (conv(C4)) ?
    fpn_results.append(n[layer])

    P6_relu = n[layer + 'relu'] = L.ReLU(n[layer], in_place=True)

    # -------------------------------------
    # 1.2.11.2 model.fpn.top_blocks.p7
    # -------------------------------------

    layer = f'{prefix}_p7'
    n[layer] = L.Convolution(P6_relu, num_output=1024, kernel_size=3, stride=2, pad=1)

    # feature pyramid P6 append at end
    # P7 = (conv(P6))
    fpn_results.append(n[layer])

    # fpn_results = [P2, P3, P4, P6, P7]
    # P2 shape: (1, 1024, 60, 72)
    # P3 shape: (1, 1024, 30, 36)
    # P4 shape: (1, 1024, 15, 18)
    # P6 shape: (1, 1024,  8,  9)
    # P7 shape: (1, 1024,  4,  5)


    # ************************************************
    # 2 model.rpn
    # rpn := head
    # head := cls_tower  => cls_logits
    #       bbox_tower => bbox_pred
    # ************************************************
    logits = []
    bbox_reg = []
    for idx, p_num in enumerate([2,3,4,6,7]):

        bottom = fpn_results[idx]

        # cls_tower + cls_logits
        prefix = f"rpn_head_cls_tower_p{p_num}"
        cls_tower_conv1 = f'{prefix}_0'
        cls_tower_relu1 = f'{prefix}_1'
        cls_tower_conv2 = f'{prefix}_2'
        cls_tower_relu2 = f'{prefix}_3'
        cls_tower_conv3 = f'{prefix}_4'
        cls_tower_relu3 = f'{prefix}_5'
        cls_tower_conv4 = f'{prefix}_6'
        cls_tower_relu4 = f'{prefix}_7'
        cls_logits = f'rpn_head_cls_logits_p{p_num}'

        n[cls_tower_conv1], n[cls_tower_relu1], n[cls_tower_conv2], n[cls_tower_relu2], \
        n[cls_tower_conv3], n[cls_tower_relu3], n[cls_tower_conv4], n[cls_tower_relu4], \
        n[cls_logits] = cls_tower_logits(bottom=bottom, nout=9, kernel_size=3, stride=1, pad =1 )

        logits.append(n[cls_logits])

        # bbox_tower + bbox_pred
        prefix = f"rpn_head_bbox_tower_p{p_num}"
        bbox_tower_conv1 = f'{prefix}_0'
        bbox_tower_relu1 = f'{prefix}_1'
        bbox_tower_conv2 = f'{prefix}_2'
        bbox_tower_relu2 = f'{prefix}_3'
        bbox_tower_conv3 = f'{prefix}_4'
        bbox_tower_relu3 = f'{prefix}_5'
        bbox_tower_conv4 = f'{prefix}_6'
        bbox_tower_relu4 = f'{prefix}_7'
        bbox_pred = f'rpn_head_bbox_pred_p{p_num}'

        n[bbox_tower_conv1], n[bbox_tower_relu1], n[bbox_tower_conv2], n[bbox_tower_relu2], \
        n[bbox_tower_conv3], n[bbox_tower_relu3], n[bbox_tower_conv4], n[bbox_tower_relu4], \
        n[bbox_pred] = bbox_tower_pred(bottom=bottom, nout=36, kernel_size=3, stride=1, pad=1)

        bbox_reg.append(n[bbox_pred])

    return n.to_proto()



def pytorch_model_load():

    from maskrcnn_benchmark.config import cfg

    # detection model conf and weight file names
    detect_model = {
        "v1":
            {
                "config_file": "config_det_v1_200723_001_180k.yaml",
                "weight_file": "model_det_v1_200723_001_180k.pth"

            },
        "v2":
            {
                "config_file": "config_det_v2_200924_002_180k.yaml",
                "weight_file": "model_det_v2_200924_002_180k.pth"
            }
    }

    # model version
    version = "v2"

    # test image file path
    image_file_path = "./sample_images/detection/1594202471809.jpg"

    config_file = os.path.join('./model/detection', detect_model[version]["config_file"])
    weight_file = os.path.join('./model/detection', detect_model[version]["weight_file"])

    is_recognition = False
    # clone project level config and merge with experiment config
    cfg = cfg.clone()
    cfg.merge_from_file(config_file)

    cfg = cfg.clone()
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model.to(device)

    # set to evaluation mode for interference
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir='/dev/null')
    _ = checkpointer.load(weight_file)

    # build_transforms defined in maskrcnn_benchmark.data.transforms/*.py
    transforms = build_transforms(cfg, is_recognition)
    cpu_device = torch.device("cpu")
    score_thresh = cfg.TEST.SCORE_THRESHOLD
    return model, transforms


def load_weights_and_biases(caffe_network, pytorch_network):
    # print pytorch_network blobs

    # print caffe_network blobs : outputs of every layers
    """
    print(f"len(caffe_network.blobs: {len(caffe_network.blobs)}")
    for k, v in caffe_network.blobs.items():
        print(k, v.data.shape)
    """

    # print caffe_network layers

    # print caffe_network params : learnable parameter (wights, biases)
    """
    print(f"len(caffe_network.params): {len(caffe_network.params)}")
    for k, v in caffe_network.params.items():
        # v[0] : weights
        # v[1] : biases
        print(f"layer_name: {k} ")
        for idx, value in enumerate(v):
            print(f"\tv[{idx}].data.shape: {value.data.shape}")

        #print(f"\tweights.shape:{v[0].data.shape}")
        #print(f"\tbias.shape: {v[1].data.shape}")
    """
    pass


def logits_and_bbox_regs(network):
    pass

def anchor_generate(image_list, feature_maps):
    pass

def box_generate(anchors, objectness, rpn_box_regression):
    pass


if __name__ == "__main__":

    #------------------------------------------------------
    # 1. detection network v2 spec preparation
    #------------------------------------------------------
    network_spec = caffe.NetSpec()

    # input pil image:  438, 512 (H, W)
    # after transform:  480, 561 (H, W) - keep aspect ratio resize
    # after to_image_list(): 480, 576  - 32 divisable with zero padding
    network_spec.data = L.DummyData(shape=[dict(dim=[1, 3, 480, 576])])

    #------------------------------------------------------
    # 2. detection network v2 spec prototxt file generation
    #------------------------------------------------------
    network_spec_proto = detection_network_v2_spec(network_spec, network_spec.data)

    with open(prototxt_file, 'w') as f:
        f.write(str(network_spec_proto))
    print(f"{prototxt_file} written ....")

    # ------------------------------------------------------
    # detection network v2 model for test (for prediction)
    # ------------------------------------------------------
    caffe_network = caffe.Net(prototxt_file, caffe.TEST)

    """
    print("==============================")
    print("==============================")
    print(f"type(caffe_network.layers): {type(caffe_network.layers)}")
    for layer in caffe_network.layers:
        print(f" of type {layer.type}")
    print("==============================")
    print("==============================")
    """

    # OrderedDict( [(name,  layer object reference) ,... , (name, layer obejct reference)])
    print(f"type(caffe_network.layer_dict): {type(caffe_network.layer_dict)}")
    """
    print(f"caffe_network.layer_dict: {caffe_network.layer_dict}")
    """

    for idx, layer_name in enumerate(caffe_network.layer_dict):
        print(f"-----------------------------")
        print(f"layer index: {idx}")
        print(f"-----------------------------")
        print(f"layer name: {layer_name}")
        print(f"layer type: {caffe_network.layer_dict[layer_name].type}")
        print(f"blobs:")
        for blob in caffe_network.layer_dict[layer_name].blobs:
            print(f"\t{blob}")
            print(f"\tblob.channels: {blob.channels}")
            print(f"\tblob.data.shape: {blob.data.shape}")
            print("")

    #------------------------------------------------------
    # load pytorch model for detection model v2
    #------------------------------------------------------
    pytorch_network, pytorch_transforms = pytorch_model_load()


    #------------------------------------------------------
    # set learnable parameters (weights and biases) in the network
    #------------------------------------------------------
    load_weights_and_biases(caffe_network, pytorch_network)

    #------------------------------------------------------
    # get input image and preprocessing
    #------------------------------------------------------

    #------------------------------------------------------
    # to_image_list
    #------------------------------------------------------

    #------------------------------------------------------
    # set input to network
    #------------------------------------------------------



    # call forward to get lostigs and bbox_regs
    #network.forward()

    # ------------------------------------------------------
    # extract logits, and bbox_regs from network
    # ------------------------------------------------------
    # logits:
    #       rpn_head_cls_logits_p2,
    #       rpn_head_cls_logits_p3,
    #       rpn_head_cls_logits_p4,
    #       rpn_head_cls_logits_p6,
    #       rpn_head_cls_logits_p7,
    # bbox_regs:
    #       rpn_head_cls_logits_p2,
    #       rpn_head_cls_logits_p3,
    #       rpn_head_cls_logits_p4,
    #       rpn_head_cls_logits_p6,
    #       rpn_head_cls_logits_p7,
    #objectness, rpn_box_regression = logits_and_bbox_regs(network)


    # ------------------------------------------------------
    # anchor generate
    # refer to anchor_generator.py : Anchor_Generator.forward()
    # ------------------------------------------------------
    #anchors = anchor_generate(image_list, feature_maps)

    # ------------------------------------------------------
    # box generate
    # refer to rpn.py : RPN_MODULE._forward_test()
    # ------------------------------------------------------
    #box_generate(anchors, objectness, rpn_box_regression)
