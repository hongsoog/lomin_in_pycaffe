# Introduction
repository for keeping source codes for FAIR Mask RCNN conversion to Caffe

# Confluence Pages
* [Lomin OCR KB](http://echo.etri.re.kr:8090/display/~kimkk/Lomin-OCR+KB)
* [Caffe Conversion KB](http://echo.etri.re.kr:8090/display/~kimkk/Caffe+Conversion+KB)
* [Lomin OCR 작업 일지](http://echo.etri.re.kr:8090/pages/viewpage.action?pageId=78086779)
  * [How to setup Lomin OCR](http://echo.etri.re.kr:8090/pages/viewpage.action?pageId=78086371#LominOCR%ED%85%8C%EC%8A%A4%ED%8A%B8%ED%99%98%EA%B2%BD%EC%84%A4%EC%B9%98-setupforLomin-OCR)

# main source files

> `./caffe_detection_v2_model.py`
*  detection v2 model in caffe implementaion

> `./caffe_detection_v2_model.ipynb`
* jupyter lab file for line by line testing codes in `./caffe_detection_v2_model.py`

> `./pytorch_detection_v2_model.ipynb`
* jupyter lab file for generating files which contains learnable parameters in PyTorch Detection V2 model
  
# Saving and Loading learnable parameters between PyTorch and Caffe

Each learnable parameter is saved into separated numpy file with file naming convention used in caffe Detection Model V2
  * refer to [./detection_v2_prototxt](https://github.com/hongsoog/lomin_in_pycaffe/blob/master/detection_v2.prototxt)

> In pytorch, model and submodel and layer name is concatenated with dot. ex.) model.backbone.body.stem.conv1
> In caffe, learnable parameters name is concatednated with unders score. ex.) model_backbone_bondy_stem_conv1

Hence file name for learnable paramter is `model_backbone_bondy_stem_conv1.npy`
