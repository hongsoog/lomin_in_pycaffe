
import sys
import inspect
from pathlib import Path
import numpy as np
import math

# pycaffe modules
import caffe
from caffe import layers as L, params as P

# PIL modules
from PIL import Image, ImageDraw

# PyPlot
import matplotlib.pyplot as plt

NPY_ROOT = f"./npy_save"

# ===============================================
# Input Image Transform Functions
# ===============================================
def get_size(pil_image, mode='keep_ratio'):
    """Return suitable size for detection or recognition model
       mask rcnn transforms.py get_size() replica

    Args:
        pil_image (PIL.Image) : PIL image opened with RGB mode (512x438=WxH)
        mode (str) : resize mode
            "horizontal_padding" | 'keep_ratio'

    Returns:
        (int, int) : tuple of (width, height) for resizing
    """
    # get size of pil_image
    w, h = pil_image.size

    # output image width and height initialize
    ow, oh = -1, -1

    # i) recognition model: 'horizontal_padding'
    if (mode == 'horizontal_padding'):
        ow, oh = -1, -1
        target_width = int(w * (oh / h))
        if target_width < oh:
            target_width = oh

        if target_width > ow:
            target_width = ow

        ow = target_width

    # ii) detection model: 'keep_ratio'
    elif (mode == 'keep_ratio'):
        min_size = 480
        max_size = 640
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))

        # summary
        # take smaller one from height or width, and resize smaller one to 480
        # and larger one is resized while keeping ratio

        # i) first determine max_size
        #   max_size : min_size  = max_original_size : min_original_size  -- (1)
        #       ? :  480  =  512 : 438
        # from (1) max_size = max_original_size * min_size / min_orignal_size
        #                   =  480*512/438 = 531.09 = 561
        # max size= 561.095
        calc_max_size = max_original_size / min_original_size * min_size
        max_size = min(calc_max_size, max_size)

        # ii) determine min_size from the determined max_size
        #   max_size : min_size  = max_original_size : min_original_size  -- (2)
        #      561.095  :  ?  =  512 : 438
        # from (2) min_size  =  max_size * min_original_size /  max_original_size
        #                    = 561.095 * 438 /512 = 479.99 = round(479.99) = 480
        min_size = round(max_size * min_original_size / max_original_size)

        # if input image is a vertical image, i.e, height > width
        #   ow = min_size, oh = max_size
        # if input image is a horizontal image, i.e, width > height
        #   ow = max_size, oh = min_size
        ow = min_size if w < h else max_size

        oh = max_size if w < h else min_size

        # oh : 480, ow = 561.095
        # int() cause round off
        # oh : 480, ow = 561,
        # (438, 512)  => (480, 561)   ; keep ratio = 1.168

    # return target size with WXH format
    return (int(ow), int(oh))


def pil_image_resize(pil_image):
    """Returns resized the PIL image using BILINEAR interpolation

    Args:
        pil_image (PIL.Image) : original PIL image to be resized
            ex. 512x438 => 561 x 480

    Returns:
        PIL.Image : resized PIL Image
    """
    # cacl height and width after resize
    # size = get_size(pil_image, mode='keep_ratio')
    width, height = get_size(pil_image, mode='keep_ratio')

    # do resize with BILINEAR interpolaton
    # resized_pil_image = Image.resize(size, resample=Image.BILINEAR)
    resized_pil_image = pil_image.resize((width, height), resample=Image.BILINEAR)

    # WxH = 512x438 RGB ==> WxH = 561x480 RGB
    return resized_pil_image


def pil_to_ndarray(pil_image):
    """Returns numpy.ndarray converted from PIL image of RGB mode

    Args:
    pil_image (PIL.Image):  resized PIL image, 561x480 (WxH), mode=RGB

    Returns:
        ndarray of float32 : array with shape of CxHxW with pixel value range 0.0 ~ 1.0
    """
    # read PIL image into np.ndarray
    image_array = np.array(pil_image)

    # pil_image.size : (w, h)
    # pil_image.mode : "RGB", len(pil_image.mode) =3
    w, h = pil_image.size
    c = len(pil_image.mode)

    # reshape 707840 into HWC (480, 561, 3) format
    image_array = image_array.reshape(h, w, c)

    # change dimension order from HWC to CHW format
    image_array = image_array.transpose(2, 0, 1)

    # change pixel value range 0 - 255 to 0.0 ~ 1.0
    image_array = np.float32(image_array) / 255.0

    return image_array


def normalize(image_array):
    """ Returns normalized ndarray of BGR ch. order
        with configuration defined mean and std

    Args:
        image_array (np.ndarray) : array format of resized input image, RGB mode and CHW dimension order

    Returns:
       ndarray: normalized with configuration defined mean and std
           dimension order: CHW, channel order: BGR
    """
    mean = [102.9801, 115.9465, 122.7717]
    std = [1.0, 1.0, 1.0]

    # change ch. order from RGB to BGR
    # https://note.nkmk.me/en/python-opencv-bgr-rgb-cvtcolor/
    image_array = image_array[[2, 1, 0], :, :]

    # multiply 255 to each pixel value
    image_array = image_array * 255

    # normalize with mean and std
    # since std is [1.0, 1.0, 1.0], just subtract mean
    # note that CHW shape with channel order BGR
    image_array[0, :, :] = image_array[0, :, :] - mean[0]
    image_array[1, :, :] = image_array[1, :, :] - mean[1]
    image_array[2, :, :] = image_array[2, :, :] - mean[2]

    return image_array

def image_preprocess(img_file_path, debug=False):
    """ Returns transformed and zero padded batch array

    Args:
        img_file_path (string): file path to image file
        debug (bool): print debug message or not
    Retruns:
        ndarray of (1, C, W, H)
    """
    # 1 test image laoding into PIL.Image with RGB mode
    img_file = "../sample_images/detection/1594202471809.jpg"
    pil_img = Image.open(img_file).convert('RGB')
    if debug:
        print(f"pil_img.size: {pil_img.size}")

    # 2 resize pil image
    resized_pil_img = pil_image_resize(pil_img)
    if debug:
        print(f"resized_pil_img.size (WxH): {resized_pil_img.size}\nresized_pil_img.mode: {resized_pil_img.mode}")

    # 2.3 PIL.Image to np.ndarray conversion
    img_array = pil_to_ndarray(resized_pil_img)
    if debug:
        print(f"img_array.shape (CxHxW): {img_array.shape}, ch. order in C dimension is RGB")
        print(f"img_array.dtype: {img_array.dtype}")

    # 2.4 Normalization
    # normalization with mean and std defined in configuration
    # mean: [102.9801, 115.9465, 122.7717], std: [1.0, 1.0, 1.0]
    normalized_img_array = normalize(img_array)
    if debug:
        print(f"normalized_img_array.shape (CxHxW): {normalized_img_array.shape}, ch. order in C dimension is GBR")
        print(f"normalized_img_array.dtype: {normalized_img_array.dtype}")

    # 2.5 Zero Padding
    # make zero image of width and height that ar multiple of 32 and
    # overlay the normalized_img_array on to zeo image
    # add batch dimension
    batch_img_array = zero_padding(normalized_img_array, size_divisible=32)
    if debug:
        print(f"batch_img_array.shape (NxCxHxW): {batch_img_array.shape}, ch. order in C dimension is GBR")
        print(f"batch_img_array.dtype: {batch_img_array.dtype}")

    return batch_img_array


def zero_padding(image_array, size_divisible=32):
    """Returns batched padded array with new width/height with 32 divisible and pad with zero pixels

    Args:
        image_array (np.ndarray) : image array of CHW dimension order and RGB channel order
            ndarray of shape format (3, H, W), RGB channel order
        size_divisible (int, default:32) : which multiple of width and height

    Returns:
        ndarray of shape [1, 3, H', W']:
            batched image array of shape (1, 3, H', W'), H' and W' is multiple of 32
            increase region filled with zeros and batch dimension added at axis 0
    """
    # calc size divisible new height and width
    c, h, w = image_array.shape

    new_h = int(np.ceil(h / size_divisible) * size_divisible)
    new_w = int(np.ceil(w / size_divisible) * size_divisible)

    # create black image with size divisible
    padded_image_array = np.zeros((3, new_h, new_w), dtype=np.float32)

    # overlay image_array on padded_image
    padded_image_array[:c, :h, :w] = image_array

    # add batch dimension into image_array
    # (3, H, W) => (1, 3, H, W)
    padded_image_array = np.expand_dims(padded_image_array, axis=0)

    return padded_image_array


# ===============================================
# Network Spec Build Functions
# ===============================================

# ----------------------------------------
# backbone.body (Resnet50) Building functions
# ----------------------------------------
def conv_fbn(bottom, nout, in_place_scale=False, **kwargs):

    """ Build a block for Conv -> FronzeBN
        Note that PyTorch FrozenBN is implemented using Scale layer

    Args:
        bottom: input to this conv/bn/scale block
        nout : num of oputs in Convolution Layer
        in_place: in place operation in scale layer

    Returns:
       top of conv, fbn
    """
    conv = L.Convolution(bottom, num_output=nout, bias_term=False, **kwargs)

    # https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/batch_norm.py
    # bn = L.BatchNorm(conv, use_global_stats=True, in_place=True)
    # fbn = L.Scale(conv, bias_term=True, in_place=True)
    fbn = L.Scale(conv, bias_term=True, in_place=in_place_scale)
    return conv, fbn

def resnet_stage_sublayer(layer_name, sublayer_num, net_spec, bottom, nout, downsample_branch=False, initial_stride=2,
                          in_place_scale=False, in_place_relu=False):
    """Build Basic Resnet Stage
       Resnet50 consits of following layers (aka stage)
       stage 0 (stem), stage 1, stage 2, stage 3 and stage 4

    Args:
       layer_num  : layer number as ? in "backbone_body_layer?"
       sub_num    : sublayer number as # in "backbone_body_layer?_#"
       net_spec   : caffe.NetSpec object
       bottom     : input to this residual block
       nout       : num. of out ch.
       downsample :  inclusion of downsample path at beginning of this residula block?
                     if BottleNeck architecutre used, set True (default: False)\

       initial_stride: stride used in branch2a and branc2b. note that branch3b stride is always 1.


    Overview diagram:
             +------> [ *_downsample_0 -- *_downsample_1 ]--------------------+
             |                                                                |
    bottom --+---------> [ *_conv1 --- *_bn1 --- *_relu1 ] ----+              |
             |                                                 |              |
             |     +-------------------------------------------+              |
             |     |                                                          |
             |     +---> [ *_conv2 --- *_bn2 --- *_relu2 ] ----+              |
             |                                                 |              | if downsample == True
             |     +-------------------------------------------+              |
             |     |                                                          V
             |     +---> [ *_conv3 --- *_bn3 --------------------------> [ *_res: Eltwise ] ----> *__res_relu :ReLu
             |                                                                 ^
             |                                                                 |
             |                                                                 | if downsample == False
             +-----------------------------------------------------------------+
    """
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
        #downsample_scale = f'{prefix}_downsample_scale'
        net_spec[downsample_conv], net_spec[downsample_bn]  = \
            conv_fbn( bottom, 4*nout, in_place_scale=in_place_scale,
                      kernel_size=1, stride=initial_stride, pad=0)
    else:
        initial_stride = 1

    # ------------------------------------------------------------------
    # In pytorch model,
    # backbone.body.layer{layer_num}.{sublayer_num}.conv1
    # backbone.body.layer{layer_num}.{sublayer_num}.bn1
    # backbone.body.layer{layer_num}.{sublayer_num}.relu1
    # ------------------------------------------------------------------
    conv1 = f'{prefix}_conv1'
    bn1 = f'{prefix}_bn1'

    net_spec[conv1], net_spec[bn1] = \
        conv_fbn( bottom, nout, in_place_scale=in_place_scale,
                  kernel_size=1, stride=initial_stride, pad=0)

    relu1 = f'{prefix}_relu1'
    net_spec[relu1] = L.ReLU(net_spec[bn1], in_place=in_place_relu)

    # ------------------------------------------------------------------
    # In pytorch model,
    # backbone.body.layer{layer_num}.{sublayer_num}.conv2
    # backbone.body.layer{layer_num}.{sublayer_num}.bn2
    # backbone.body.layer{layer_num}.{sublayer_num}.relu2
    # ------------------------------------------------------------------
    conv2 = f'{prefix}_conv2'
    bn2 = f'{prefix}_bn2'

    net_spec[conv2], net_spec[bn2] = \
        conv_fbn( net_spec[relu1], nout, in_place_scale=in_place_scale,
                  kernel_size=3, stride=1, pad=1)

    relu2 = f'{prefix}_relu2'
    net_spec[relu2] = L.ReLU(net_spec[bn2], in_place=in_place_relu)

    # ------------------------------------------------------------------
    # In pytorch model,
    # backbone.body.layer{layer_num}.{sublayer_num}.conv3
    # backbone.body.layer{layer_num}.{sublayer_num}.bn3
    # note that no relu after bn3 !!
    # ------------------------------------------------------------------
    conv3 = f'{prefix}_conv3'
    bn3 = f'{prefix}_bn3'

    net_spec[conv3], net_spec[bn3] = \
        conv_fbn( net_spec[relu2], 4 * nout, in_place_scale=in_place_scale,
                  kernel_size=1, stride=1, pad=0)

    # ---------------------------------------
    # skip connection processing
    # ---------------------------------------
    eltwise = f'{prefix}_eltwise'

    if downsample_branch:
        net_spec[eltwise] = L.Eltwise(net_spec[downsample_bn], net_spec[bn3])
    else:
        net_spec[eltwise] = L.Eltwise(bottom, net_spec[bn3])

    # ---------------------------------------
    # last relu
    # backbone.body.layer{layer_num}.{sublayer_num}.relu
    # ---------------------------------------
    relu = f'{prefix}_relu'
    # n[relu] = L.ReLU(n[eltwise], in_place=True)
    net_spec[relu] = L.ReLU(net_spec[eltwise], in_place=in_place_relu)


def cls_tower_logits(bottom, nout=9, bias_term=True, **kwargs):
    '''Builds cls_tower_logits in form of [conv => relu]x4 + conv block.
       all conv layer uses kernel:3, stride: 1, pad: 1

       Params:
          bottom (blob) : input to cls tower logits
          nout (uint) : num. of outputs in cls logit (last conv layer)
          bias_term (bool) : use bias term or not in conv layer
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
def detection_network_v2_spec(net_spec, bottom, in_place_scale=False, in_place_relu=False):

    """Build network spect of Backbone with ResNet50 + FPN and RPN

    Args:
         n (caffe.NetSpec) : NetSpec instance
         bottom (blob) : innput to detection network
         in_place_scale: use in_place in caffe Scale layer of fbn implemetation
         in_place_relu: use in_place in caffe ReLu layer of fbn implemetation

    Returns:
        prototxt object of network spec

    """

    # ************************************************
    # 1. model.backbone
    # ************************************************
    # backbone := body + fpn
    # ************************************************


    # =================================================================
    # 1.1 model.backbone.body : Resnet50
    # body = stem (layer 0) +  layer 1 + layer 2 + layer 3 + layer 4
    # =================================================================
    """
    Resnet50 consists of following layers (aka stage)
     * layer 0: conv + fbn + relu + maxpool
     * layer 1: 3 sublayer 0, 1, 2 with oen down_sample branch
     * layer 2: 4 sublayer 0, 1, 2, 3 with one down_sample branch
     * layer 3: 6 sublayer 0, 1, 2, 3, 4, 5 with one down_sample branch
     * layer 4: 3 sublayer 0, 1, 2 with one down_sample branch
    """
    # keep list of feature maps in stage order
    backbone_body_features = []

    prefix = "backbone_body"
    # -------------------------------------
    # 1.1.0 model.backbone.body.stem (layer0)
    # sublayer: conv1,  bn1, relu, maxpool
    # -------------------------------------
    layer_name = f'{prefix}_stem_'

    net_spec[layer_name + 'conv1'], net_spec[layer_name + 'bn1'] = \
        conv_fbn(bottom, 64, in_place_scale=in_place_scale, kernel_size=7, pad=3, stride=2)

    net_spec[layer_name + 'relu'] = L.ReLU(net_spec[layer_name + 'bn1'], in_place=in_place_relu)
    net_spec[layer_name + 'maxpool'] = L.Pooling(net_spec[layer_name + 'relu'], kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # -------------------------------------
    # 1.1.1 model.backbone.body.layer1
    # sublayer: 0, 1, 2
    # -------------------------------------
    pre_layer_name = layer_name
    layer_name = f'{prefix}_layer1'

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='0', net_spec=net_spec,
                          bottom=net_spec[pre_layer_name + 'maxpool'],
                          nout=64, downsample_branch=True, initial_stride=1,
                          in_place_scale=in_place_scale, in_place_relu=in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='1', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_0_relu'],
                          nout=64,  downsample_branch=False, initial_stride=2,
                          in_place_scale=in_place_scale, in_place_relu=in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='2', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_1_relu'],
                          nout=64, downsample_branch=False, initial_stride=2,
                          in_place_scale = in_place_scale, in_place_relu = in_place_relu)

    # feature C1
    backbone_body_features.append(net_spec[layer_name + '_2_relu'])

    # -------------------------------------
    # 1.1.2 model.backbone.body.layer2 (stage 2)
    # sublayer: 0, 1, 2, 3
    # -------------------------------------
    pre_layer_name = layer_name
    layer_name = f'{prefix}_layer2'

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='0', net_spec=net_spec,
                          bottom=net_spec[pre_layer_name + '_2_relu'],
                          nout=128, downsample_branch=True, initial_stride=2,
                          in_place_scale = in_place_scale, in_place_relu = in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='1', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_0_relu'],
                          nout=128, downsample_branch=False, initial_stride=2,
                          in_place_scale = in_place_scale, in_place_relu = in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='2', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_1_relu'],
                          nout=128, downsample_branch=False, initial_stride=2,
                          in_place_scale = in_place_scale, in_place_relu = in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='3', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_2_relu'],
                          nout=128, downsample_branch=False, initial_stride=2,
                          in_place_scale = in_place_scale, in_place_relu = in_place_relu)

    # feature C2
    backbone_body_features.append(net_spec[layer_name + '_3_relu'])

    # -------------------------------------
    # 1.1.3 model.backbone.body.layer3 (stage 3)
    # sublayer: 0, 1, 2, 3, 4, 5
    # -------------------------------------
    pre_layer_name = layer_name
    layer_name = f'{prefix}_layer3'

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='0', net_spec=net_spec,
                          bottom=net_spec[pre_layer_name + '_3_relu'],
                          nout=256, downsample_branch=True, initial_stride=2,
                          in_place_scale = in_place_scale, in_place_relu = in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='1', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_0_relu'],
                          nout=256, downsample_branch=False, initial_stride=2,
                          in_place_scale=in_place_scale, in_place_relu=in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='2', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_1_relu'],
                          nout=256, downsample_branch=False, initial_stride=2,
                          in_place_scale=in_place_scale, in_place_relu=in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='3', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_2_relu'],
                          nout=256, downsample_branch=False, initial_stride=2,
                          in_place_scale=in_place_scale, in_place_relu=in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='4', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_3_relu'],
                          nout=256, downsample_branch=False, initial_stride=2,
                          in_place_scale=in_place_scale, in_place_relu=in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='5', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_4_relu'],
                          nout=256, downsample_branch=False, initial_stride=2,
                          in_place_scale=in_place_scale, in_place_relu=in_place_relu)


    # feature C3
    backbone_body_features.append(net_spec[layer_name + '_5_relu'])

    # -------------------------------------
    # 1.1.4 model.backbone.body.layer4 (stage 4)
    # sublayer: 0, 1, 2
    # -------------------------------------
    pre_layer_name = layer_name
    layer_name = f'{prefix}_layer4'

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='0', net_spec=net_spec,
                          bottom=net_spec[pre_layer_name + '_5_relu'],
                          nout=512, downsample_branch=True, initial_stride=2,
                          in_place_scale = in_place_scale, in_place_relu = in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='1', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_0_relu'],
                          nout=512, downsample_branch=False, initial_stride=2,
                          in_place_scale=in_place_scale, in_place_relu=in_place_relu)

    resnet_stage_sublayer(layer_name=layer_name, sublayer_num='2', net_spec=net_spec,
                          bottom=net_spec[layer_name + '_1_relu'],
                          nout=512, downsample_branch=False, initial_stride=2,
                          in_place_scale=in_place_scale, in_place_relu=in_place_relu)

    # feature C4
    backbone_body_features.append(net_spec[layer_name + '_2_relu'])

    C1, C2, C3, C4 = backbone_body_features

    # =================================================================
    # 1.2 model.backbone.fpn
    # =================================================================
    backbone_fpn_features = []
    prefix = "backbone_fpn"

    # -------------------------------------
    # 1.2.1 model.backbone.fpn.fpn_inner4
    # -------------------------------------
    layer_name = f'{prefix}_fpn_inner4'
    last_inner = net_spec[layer_name] = L.Convolution(C4, num_output=1024, kernel_size=1, stride=1)

    # -------------------------------------
    # 1.2.2 model.backbone.fpn.fpn_layer4
    # -------------------------------------
    layer_name = f'{prefix}_fpn_layer4'
    net_spec[layer_name] = L.Convolution(last_inner, num_output=1024, kernel_size=3, stride=1, pad=1)

    # featuer pyramid feature P4 append
    # P4 = conv (conv (C4))
    backbone_fpn_features.append(net_spec[layer_name])

    # -------------------------------------
    # 1.2.3 model.backbone.fpn_inner3_upsample
    # -------------------------------------
    layer_name = f'{prefix}_fpn_inner3_upsample' # inner3__upsample
    inner3_upsample = net_spec[layer_name] = \
        L.Deconvolution(last_inner, convolution_param = dict(num_output=1024,
                                                             kernel_size=4,
                                                             stride=2, pad=1,
                                                             weight_filler=dict(type ='bilinear'),
                                                             bias_term=False) )

    # -------------------------------------
    # 1.2.4 model.backbone.fpn_inner3_lateral
    # -------------------------------------
    #layer_name = f'{prefix}_fpn_inner3_lateral'
    layer_name = f'{prefix}_fpn_inner3'
    inner3_lateral = net_spec[layer_name] = L.Convolution(C3, num_output=1024, kernel_size=1, stride=1)

    # -------------------------------------
    # 1.2.5 model.backbone.fpn_inner3_lateral_sum
    # -------------------------------------
    layer_name = f'{prefix}_fpn_inner3_lateral_sum' # P3
    last_inner = net_spec[layer_name] = L.Eltwise(inner3_lateral,  inner3_upsample)

    # -------------------------------------
    # 1.2.6 model.backbone.fpn_layer3
    # -------------------------------------
    layer_name = f'{prefix}_fpn_layer3'
    net_spec[layer_name] = L.Convolution(last_inner, num_output=1024, kernel_size=3, stride=1, pad=1)

    # feature pyramid P3 insert at idx 0
    # P3 = unsample(P4) + conv(conv(C3))
    backbone_fpn_features.insert(0, net_spec[layer_name])

    # -------------------------------------
    # 1.2.7 model.backbone.fpn_inner2_upsample
    # -------------------------------------
    layer_name = f'{prefix}_fpn_inner2_upsample' # inner2__upsample
    inner2_upsample = net_spec[layer_name] = \
        L.Deconvolution(last_inner, convolution_param = dict(num_output=1024,
                                                             kernel_size=4,
                                                             stride=2, pad=1,
                                                             weight_filler=dict(type ='bilinear'),
                                                             bias_term=False))

    # -------------------------------------
    # 1.2.8 model.backbone.fpn_inner2_lateral
    # -------------------------------------
    #layer_name = f'{prefix}_fpn_inner2_lateral'
    layer_name = f'{prefix}_fpn_inner2'
    inner2_lateral = net_spec[layer_name] = L.Convolution(C2, num_output=1024, kernel_size=1, stride=1)

    # -------------------------------------
    # 1.2.9 model.backbone.fpn_inner2_lateral_sum
    # -------------------------------------
    layer_name = f'{prefix}_fpn_inner2_lateral_sum' # P3
    last_inner = net_spec[layer_name] = L.Eltwise(inner2_lateral,  inner2_upsample)

    # -------------------------------------
    # 1.2.10 model.backbone.fpn_layer2
    # -------------------------------------
    layer_name = f'{prefix}_fpn_layer2'
    net_spec[layer_name] = L.Convolution(last_inner, num_output=1024, kernel_size=3, stride=1, pad=1)

    # feature pyramid P2 insert at idx 0
    # P2 = unsample(P3) + conv(conv(C2))
    backbone_fpn_features.insert(0, net_spec[layer_name])


    # -------------------------------------
    # 1.2.11 model.fpn.top_blocks
    # -------------------------------------

    # -------------------------------------
    # 1.2.11.1 model.fpn.top_blocks.p6
    # -------------------------------------
    prefix = "backbone_fpn_top_blocks"
    layer_name = f'{prefix}_p6'
    net_spec[layer_name] = L.Convolution(C4, num_output=1024, kernel_size=3, stride=2, pad=1)

    # feature pyramid P6 append at end
    # P6 = (conv(C4)) ?
    backbone_fpn_features.append(net_spec[layer_name])

    P6_relu = net_spec[layer_name + 'relu'] = L.ReLU(net_spec[layer_name], in_place=True)

    # -------------------------------------
    # 1.2.11.2 model.fpn.top_blocks.p7
    # -------------------------------------

    layer_name = f'{prefix}_p7'
    net_spec[layer_name] = L.Convolution(P6_relu, num_output=1024, kernel_size=3, stride=2, pad=1)

    # feature pyramid P6 append at end
    # P7 = (conv(P6))
    backbone_fpn_features.append(net_spec[layer_name])

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

        bottom = backbone_fpn_features[idx]

        # cls_tower + cls_logits
        #prefix = f"rpn_head_cls_tower_p{p_num}"
        prefix = f"rpn_head_cls_tower"
        suffix = f"for_p{p_num}"
        cls_tower_conv1 = f'{prefix}_0_{suffix}'
        cls_tower_relu1 = f'{prefix}_1_{suffix}'
        cls_tower_conv2 = f'{prefix}_2_{suffix}'
        cls_tower_relu2 = f'{prefix}_3_{suffix}'
        cls_tower_conv3 = f'{prefix}_4_{suffix}'
        cls_tower_relu3 = f'{prefix}_5_{suffix}'
        cls_tower_conv4 = f'{prefix}_6_{suffix}'
        cls_tower_relu4 = f'{prefix}_7_{suffix}'
        cls_logits = f'rpn_head_cls_logits_{suffix}'

        net_spec[cls_tower_conv1], net_spec[cls_tower_relu1], net_spec[cls_tower_conv2], \
        net_spec[cls_tower_relu2], net_spec[cls_tower_conv3], net_spec[cls_tower_relu3], \
        net_spec[cls_tower_conv4], net_spec[cls_tower_relu4], net_spec[cls_logits] = \
            cls_tower_logits(bottom=bottom, nout=9, kernel_size=3, stride=1, pad =1 )

        logits.append(net_spec[cls_logits])

        # bbox_tower + bbox_pred
        #prefix = f"rpn_head_bbox_tower_p{p_num}"
        prefix = f"rpn_head_bbox_tower"
        suffix = f"for_p{p_num}"
        bbox_tower_conv1 = f'{prefix}_0_{suffix}'
        bbox_tower_relu1 = f'{prefix}_1_{suffix}'
        bbox_tower_conv2 = f'{prefix}_2_{suffix}'
        bbox_tower_relu2 = f'{prefix}_3_{suffix}'
        bbox_tower_conv3 = f'{prefix}_4_{suffix}'
        bbox_tower_relu3 = f'{prefix}_5_{suffix}'
        bbox_tower_conv4 = f'{prefix}_6_{suffix}'
        bbox_tower_relu4 = f'{prefix}_7_{suffix}'
        bbox_pred = f'rpn_head_bbox_pred_{suffix}'

        net_spec[bbox_tower_conv1], net_spec[bbox_tower_relu1], net_spec[bbox_tower_conv2], \
        net_spec[bbox_tower_relu2], net_spec[bbox_tower_conv3], net_spec[bbox_tower_relu3], \
        net_spec[bbox_tower_conv4], net_spec[bbox_tower_relu4], net_spec[bbox_pred] = \
            bbox_tower_pred(bottom=bottom, nout=36, kernel_size=3, stride=1, pad=1)

        bbox_reg.append(net_spec[bbox_pred])

    return net_spec.to_proto()


# ---------------------------------------
# network parameter loading functions
# ---------------------------------------
def load_backbone_body_params(network, debug=False):

    """Load backbone body (resnet50) parameters

    Args:
        network (caffe.Net): Network instance
        debug (bool): print debug message or not

    Returns:
        network with backbone body paramters filled

    """
    # log file patth: ./load_backbone_body_parms_log.txt
    if debug:
        my_name = inspect.currentframe().f_code.co_name
        log_file_path = f"./log/{my_name}_log.txt"
        original_std_out = sys.stdout
        f = open(log_file_path, 'w')
        sys.stdout = f

    # assumption:
    # backbone body (resnet50) learnable layer parameters npy file name start with 'backbone_body_' prefix
    k_list = [k for k in network.params.keys() if k.startswith('backbone_body_') and "data" not in k]
    total_num_of_params = len(k_list)
    num_processing = 1

    # learnable layers' weight/bias parameters distinguished by suffix 'weight', 'bias'
    suffix = ["weight", "bias"]
    # num_layers = len(network.layer_dict)

    for idx, layer_name in enumerate(network.layer_dict):

        if layer_name in k_list:
            print(f"\n-----------------------------")
            # print(f"layer index: {idx}/{num_layers}")
            print(f"Processing {num_processing}/{total_num_of_params}")
            print(f"layer name: '{layer_name}''")
            print(f"layer type: '{network.layers[idx].type}'")

            params = network.params[layer_name]
            print(f"{len(params)} learnable parameters in '{network.layers[idx].type}' type")

            for i, p in enumerate(params):
                # print(f"\tparams[{i}]: {p}")
                # print(f"\tparams[{i}] CxHxW: {p.channels}x{p.height}x{p.width}")
                print(f"\tp[{i}]: {p.data.shape} of {p.data.dtype}")

                param_file_path = f"{NPY_ROOT}/{layer_name}_{suffix[i]}.npy"

                param_file = Path(param_file_path)
                if param_file.exists():
                    print(f"\tload {param_file_path}")
                    arr = np.load(param_file_path, allow_pickle=True)

                    if p.data.shape == arr.shape:
                        print(f"\tset {layer_name}_{suffix[i]} with arr:shape {arr.shape}, type {arr.dtype}")

                        p.data[...] = arr

                        if not (np.allclose(p.data, arr)):
                            print(f"\t>>>>>> p.data is not euqal to arr")
                            print(f"\tp.data: {p.data}")
                            print(f"\tarr: {arr}")

                            if debug:
                                sys.stdout = original_std_out
                                f.close()

                            return False
                    else:
                        print(f">>>>>> p.data.shape: {p.data.shape} is not equal to arr.shape: {arr.shape}")
                        if debug:
                            sys.stdout = original_std_out
                            f.close()

                        return False
                else:
                    print(f">>>>>> {param_file_path} is not exits!!")
                    if debug:
                        sys.stdout = original_std_out
                        f.close()

                    return False
                    # END for i, pin in enumerate(params):

            num_processing += 1

    # END for idx, layer_name in enumerate(network.layer_dict):

    print(f"success!!")
    if debug:
        sys.stdout = original_std_out
        f.close()
    return True


def load_backbone_fpn_params(network, debug=False):
    """Load backbone fpn (feature pyramid network) parameters

    Args:
        network (caffe.Net): Network instance
        debug (bool): print debug message or not

    Returns:
        network with backbone fpn paramters filled

    """
    # log file patth: ./load_backbone_fpn_parms_log.txt
    if debug:
        # get current function name
        # https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function-without-using-traceback
        # my_name =  inspect.stack()[0][3]
        my_name = inspect.currentframe().f_code.co_name
        log_file_path = f".{my_name}_log.txt"
        original_std_out = sys.stdout
        f = open(log_file_path, 'w')
        sys.stdout = f
    else:
        sys.stdout = sys.stdout

    # assumption:
    # backbone fpn learnable layer parameters npy file name start with 'backbone_fpn' prefix
    # upsample layer is implmented with Decov layer in caffe, hence no parameter loading
    k_list = [k for k in network.params.keys() if k.startswith('backbone_fpn') and "upsample" not in k]
    total_num_of_params = len(k_list)
    num_processing = 1

    # learnable layers' weight/bias parameters distinguished by suffix 'weight', 'bias'
    suffix = ["weight", "bias"]
    num_layers = len(network.layer_dict)

    for idx, layer_name in enumerate(network.layer_dict):

        if layer_name in k_list:
            print(f"\n-----------------------------")
            # print(f"layer index: {idx}/{num_layers}")
            print(f"Processing {num_processing}/{total_num_of_params}")
            print(f"layer name: '{layer_name}''")
            print(f"layer type: '{network.layers[idx].type}'")

            params = network.params[layer_name]
            print(f"{len(params)} learnable parameters in '{network.layers[idx].type}' type")

            for i, p in enumerate(params):
                # print(f"\tparams[{i}]: {p}")
                # print(f"\tparams[{i}] CxHxW: {p.channels}x{p.height}x{p.width}")
                print(f"\tp[{i}]: {p.data.shape} of {p.data.dtype}")

                param_file_path = f"../npy_save/{layer_name}_{suffix[i]}.npy"

                param_file = Path(param_file_path)
                if param_file.exists():
                    print(f"\tload {param_file_path}")
                    arr = np.load(param_file_path, allow_pickle=True)

                    if p.data.shape == arr.shape:
                        print(f"\tset {layer_name}_{suffix[i]} with arr:shape {arr.shape}, type {arr.dtype}")

                        p.data[...] = arr
                        if not (np.allclose(p.data, arr)):
                            print(f"\t>>>>>> p.data is not euqal to arr")
                            print(f"\tp.data: {p.data}")
                            print(f"\tarr: {arr}")

                            if debug:
                                sys.stdout = original_std_out
                                f.close()
                            return False

                    else:
                        print(f">>>>>> p.data.shape: {p.data.shape} is not equal to arr.shape: {arr.shape}")
                        if debug:
                            sys.stdout = original_std_out
                            f.close()

                        return False
                else:
                    print(f">>>>>> {param_file_path} is not exits!!")
                    if debug:
                        sys.stdout = original_std_out
                        f.close()

                    return False
                    # END for i, p in enumearte(params)
            num_processing += 1

    # END for idx, layer_name in enumerate(network.layer_dict):
    if debug:
        sys.stdout = original_std_out
        f.close()

    return True


def load_rpn_params(network, debug=False):
    """Load rpn (region proposal network) parameters

    Args:
        network (caffe.Net): Network instance
        debug (bool): print debug message or not

    Returns:
        network with rpn paramters filled

    """
    # log file patth: ./load_rpn_params_log.txt
    if debug:
        # get current function name
        # https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function-without-using-traceback
        # my_name =  inspect.stack()[0][3]
        my_name = inspect.currentframe().f_code.co_name
        log_file_path = f"./{my_name}_log.txt"
        original_std_out = sys.stdout
        f = open(log_file_path, 'w')
        sys.stdout = f
    else:
        sys.stdout = sys.stdout

    # assumption:
    # backbone rpn learnable layer parameters npy file name start with 'rpn_head' prefix
    k_list = [k for k in network.params.keys() if k.startswith('rpn_head')]
    total_num_of_params = len(k_list)
    num_processing = 1

    suffix = ["weight", "bias"]
    # num_layers = len(network.layer_dict)

    for idx, layer_name in enumerate(network.layer_dict):

        if layer_name in k_list:
            print(f"\n-----------------------------")
            # print(f"layer index: {idx}/{num_layers}")
            print(f"Processing {num_processing}/{total_num_of_params}")
            print(f"layer name: '{layer_name}''")
            print(f"layer type: '{network.layers[idx].type}'")

            params = network.params[layer_name]
            print(f"{len(params)} learnable parameters in '{network.layers[idx].type}' type")

            for i, p in enumerate(params):
                # print(f"\tparams[{i}]: {p}")
                # print(f"\tparams[{i}] CxHxW: {p.channels}x{p.height}x{p.width}")
                print(f"\tp[{i}]: {p.data.shape} of {p.data.dtype}")

                # note remove '_for_pn' suffix from layer name
                param_file_path = f"../npy_save/{layer_name[:-7]}_{suffix[i]}.npy"

                param_file = Path(param_file_path)
                if param_file.exists():
                    print(f"\tload {param_file_path}")
                    arr = np.load(param_file_path, allow_pickle=True)

                    if p.data.shape == arr.shape:
                        print(f"\tset {layer_name}_{suffix[i]} with arr:shape {arr.shape}, type {arr.dtype}")

                        p.data[...] = arr
                        if not (np.allclose(p.data, arr)):
                            print(f"\t>>>>>> p.data is not euqal to arr")
                            print(f"\tp.data: {p.data}")
                            print(f"\tarr: {arr}")

                            if debug:
                                sys.stdout = original_std_out
                                f.close()

                            return False
                    else:
                        print(f">>>>>> p.data.shape: {p.data.shape} is not equal to arr.shape: {arr.shape}")
                        if debug:
                            sys.stdout = original_std_out
                            f.close()

                        return False
                else:
                    print(f">>>>>> {param_file_path} is not exits!!")
                    if debug:
                        sys.stdout = original_std_out
                        f.close()

                    return False
                    # END for i, p in enumearte(params)
            num_processing += 1


    # END for idx, layer_name in enumerate(network.layer_dict):
    if debug:
        sys.stdout = original_std_out
        f.close()

    return True


def load_detection_v2_parameters(detection_v2_network, debug=False):
    """load detection v2 network paramters

    Args:
        nw: caff.Net instance

    Returns:
        network with backbone body, backbone fpn and rpn paramters filled
    """
    # backbone body (resnet50) paramters loading
    if (load_backbone_body_params(network=detection_v2_network, debug=debug)):
        print(f"backbone body parameter loading success.")
    else:
        print(f"backbone body parameter loading failed.")
        return False

    # backbone fpn (feature pyramid network) paramters loading
    if (load_backbone_fpn_params(network=detection_v2_network, debug=debug)):
        print(f"backbone fpn parameter loading success.")
    else:
        print(f"backbone fpn parameter loading failed.")
        return False

    # rpn (region proposal network) paramters loading
    if (load_rpn_params(network=detection_v2_network, debug=debug)):
        print(f"rpn parameter loading success.")
    else:
        print(f"rpn parameter loading failed.")
        return False

    return True

# ---------------------------------------
# network parameter saving functions
# ---------------------------------------
def save_network_parameters(test_image_file, prototxt_file, caffemodel_file, overwrite=False):
    """load network parameters from pytorch model saved npy file
       and save into caffemodel format file
    Args:
        test_iamge_file: file path to test input image file
        prototxt_file: file path to save .prototxt file
        caffemodel_file: file path to save .caffemodel file
        overwrite (bool): overwrite existing prototxt and caffemodel file
    Returns:
        none
    """

    if not overwrite:
        # if recreate==False, then check files existence
        prototxt_file_path = Path(prototxt_file)
        caffemodel_file_path = Path(caffemodel_file)
        if prototxt_file_path.exists() and caffemodel_file_path.exists():
            # if both files already exist, then return False
            print(f"{prototxt_file} and {caffemodel_file} already exist, hence just return")
            return False

    nw_spec = caffe.NetSpec()

    # load image file and convert to batch zero padded array
    batch_img_array = image_preprocess(test_image_file, debug=True)

    # set shape of Data layer input in detection v2 model spec
    nw_spec.data = L.DummyData(shape=[dict(dim=list(batch_img_array.shape))])

    # prototxt generation of detection v2 model spec
    detection_v2_prototxt = detection_network_v2_spec(nw_spec, nw_spec.data)

    # saving prototxt spec into .prototxt file
    with open(prototxt_file, 'w') as f:
        f.write(str(detection_v2_prototxt))
    print(f"{prototxt_file} for detection v2 mode written....")

    # caffe.Net instantiation for loading params from npy files
    network = caffe.Net(prototxt_file, caffe.TEST)

    # load detection v2 network layers parameters
    load_detection_v2_parameters(network, debug=True)

    network.save(caffemodel_file)
    print(f"{caffemodel_file} for detection v2 mode written....")

    return True

if __name__ == "__main__":

    test_image_file = "../sample_images/detection/1594202471809.jpg"
    prototxt_file = "./detection_v2.prototxt"
    prototxt_file = "./detection_v2.caffemodel"

    # load network parameters from pytorch model saved npy files
    # and save paramters into '.caffemodel' file
    # Note: call this function with recreate=True only if you don't have prototxt and caffemodel file yet
    save_network_parameters(test_image_file, prototxt_file, caffemodel_file, recreate=True)

    # detection v2 network instantiation for TEST mode with prototxt_file and caffemodel file
    detection_v2_network = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)

    # set input for inference
    # load image file and convert to batch zero padded array
    batch_img_array = image_preprocess(test_image_file, debug=True)

    # set detection v2 network input
    detection_v2_network.blobs['data'].data[...] = batch_img_array

    # inference for input image
    outputs = detection_v2_network.forward()

    print(f"type(outputs): {type(outputs)}")
    for k, v in outputs.items():
        print(f"{k} of shape {v.shape}")




