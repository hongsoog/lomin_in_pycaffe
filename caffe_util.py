import numpy as np
import caffe
from caffe import layers as L, params as P
from PIL import Image, ImageDraw

#--------------------------------------------------
# Transform related function
#--------------------------------------------------
# - get_size()
# - pil_image_resize()
# - to_ndarray()
# - normalize()
#--------------------------------------------------
def get_size(pil_image, mode='keep_ratio'):
    """Returns suitable size for detection or recognition model

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
    # calculate new width/height for resizing
    # size = get_size(pil_image, mode='keep_ratio')
    width, height = get_size(pil_image, mode='keep_ratio')

    # do resize with BILINEAR inter-polaton
    # resized_pil_image = Image.resize(size, resample=Image.BILINEAR)
    resized_pil_image = pil_image.resize((width, height), resample=Image.BILINEAR)

    # WxH = 512x438 RGB ==> WxH = 561x480 RGB
    return resized_pil_image


def to_ndarray(pil_image):
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
    c = 3

    # reshape 707840 into HWC (480, 561, 3) format
    image_array = image_array.reshape(h, w, c)

    # change dimension order from HWC to CHW format
    image_array = image_array.transpose(2, 0, 1)

    # change pixel value range 0 - 255 (int) to 0.0 ~ 1.0 (float32)
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
    # note that dimension order use CHW format
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

    c, h, w = image_array.shape

    # calculate size divisible new height and width
    new_h = int(np.ceil(h / size_divisible) * size_divisible)
    new_w = int(np.ceil(w / size_divisible) * size_divisible)

    # create black image with size divisible
    # of shape (3, H', W')
    padded_image_array = np.zeros((3, new_h, new_w), dtype=np.float32)

    # overlay image_array on padded_image
    padded_image_array[:c, :h, :w] = image_array

    # add batch dimension at axis=0 into padded image array
    # (3, H', W') => (1, 3, H', W')
    padded_image_array = np.expand_dims(padded_image_array, axis=0)

    return padded_image_array



#--------------------------------------------------
# RsetNet50 (backbone) related function
#--------------------------------------------------
# - conv_bn_scale() : conv => bn => scale block creation
# - residual_block() : Resnet residual block creation
#--------------------------------------------------
def conv_bn_scale(bottom, nout, bias_term=False, **kwargs):
    '''
    build a Conv -> BN -> Scale block

        param: bottom : input to this conv/bn/scale block
        type:  bottom :
        param: nout   : num of oputs in Convolution Layer
        type:  uint
        param: bias_term : bias term used in Scale Layer
        type: bool : default False
        return: top of conv, bn, scale
        rtype:
    '''
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term, **kwargs)

    # https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/batch_norm.py
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    return conv, bn, scale



def resnet_stage_sublayer(layer_name, sublayer_num, n, bottom, nout, downsample_branch=False, initial_stride=2):
    '''Builds a single Resnet Stage

       resetne50 of model.backbone has 5 stages
       - stage 0 (stem), stage 1, stage 2, stage 3 and stage 4

    Prameters:
       layer_num (uint): layer number as ? in "backbone_body_layer?"
       sub_num (uint): sublayer number as # in "backbone_body_layer?_#"
       n (caffe.NetSpec): NetSpec object
       bottom (Blob): input to this stage sublayer
       nout (uint): num. of out ch.
       downsample (bool) :  inclusion of subsample at beginning of this stage sublayer?
                     if BottleNeck architecutre used, set True (default: False)

       initial_stride (uint, default=2): stride used in branch2a and branc2b. note that downsample(branch3b) stride is always 1.


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
          bias_trerm (bool) : use bias term or not in conv layer
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


