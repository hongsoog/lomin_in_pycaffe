#!/home/kimkk/miniconda3/envs/lomin/bin/python

import os
import argparse
from pprint import pprint
from datetime import datetime
from glob import glob
from tqdm import tqdm

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.ticker import AutoMinorLocator

import torch
import numpy as np

#from torchinfo import summary
import torchinfo

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.label_catalog import LabelCatalog
from maskrcnn_benchmark.utils.converter import Converter


class DetectionDemo(object):
    #--------------------------------
    # __init__(cfg, weight, is_recognition=False)
    #--------------------------------
    def __init__(self, cfg, weight, is_recognition=False):
        self.is_recognition = is_recognition
        self.cfg = cfg.clone()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model =  build_detection_model(self.cfg)
        self.model.to(self.device)

        # set to evaluation mode for interference
        self.model.eval()

        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir='/dev/null')
        _ = checkpointer.load(weight)

        # build_transforms defined in  maskrcnn_benchmark.data.transforms/*.py
        self.transforms = build_transforms(self.cfg, self.is_recognition)
        self.cpu_device = torch.device("cpu")
        self.score_thresh = self.cfg.TEST.SCORE_THRESHOLD

    #--------------------------------
    # run_on_pil(image_origin)
    #--------------------------------
    def run_on_pil_image(self, image_origin):

        # pil_image defined in __main__
        prediction = self.compute_prediction(pil_image)
        prediction = self.filter_by_score(prediction)
        prediction = prediction.resize(image_origin.size)
        result = self.parse_result(prediction)
        return result

    #--------------------------------
    # compute_predicion(image)
    #--------------------------------
    def compute_prediction(self, image):
        global start_time, end_time

        # convert PIL Image to torch Tensor
        image_tensor = self.transforms(image)

        image_list = to_image_list(image_tensor, self.cfg.DATALOADER.SIZE_DIVISIBILITY).to(self.device)


        """
        torchinfo.summary(model=self.model,
                          input_size=(1, 3, 480, 576),
                          input_data = image_list,
                          col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
                          verbose=2)
        """

        with torch.no_grad():

            start_time = datetime.now()
            pred = self.model(image_list)
            end_time = datetime.now()

            pred = pred[0].to(self.cpu_device)


        return pred

    #--------------------------------
    # filter_by_score(predition)
    #--------------------------------
    def filter_by_score(self, prediction):
        filter_thres = prediction.get_field('scores') > self.score_thresh
        return prediction[filter_thres]

    #--------------------------------
    # parse_result(pred)
    #--------------------------------
    def parse_result(self, pred):
        bbox = pred.bbox.numpy().tolist()
        scores = pred.get_field('scores').numpy().tolist()
        #labels = pred.get_field('labels').numpy().tolist()
        return dict(
            bboxes=bbox,     # list of [x1, y1, x2, y2]
            #labels=labels,  # no interest on label
            scores=scores,
        )


# detection model conf and weidht file names
detect_model = {
        "v1" :
        {
            "config_file" : "config_det_v1_200723_001_180k.yaml",
            "weight_file" : "model_det_v1_200723_001_180k.pth"

        },
        "v2" :
        {
            "config_file" : "config_det_v2_200924_002_180k.yaml",
            "weight_file" : "model_det_v2_200924_002_180k.pth"
        }
}

start_time = None
end_time = None

# bounding box
bb_color = 'red'
bb_line_width = 1

# bboxes score threshold
bb_score_threshold = 0.5

#----------------------------------------
# draw image with bb
#----------------------------------------
def bb_image_draw(pil_image, bb_color=(0, 0,255), bb_line_width=4, bb_score_threshold =0.5):


    # for refreshing, make a copy of pil_image
    pil_image_cp = pil_image.copy()

    # for drawing bbox
    draw = ImageDraw.Draw(pil_image_cp)


    num_bbox_included = 0

    # draw boxes with its score is larger than bb_score_threshold
    for idx, bbox in enumerate(bboxes_list):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

        if scores_list[idx] > bb_score_threshold:
                num_bbox_included += 1
                draw.rectangle( ((x1, y1), (x2, y2)), outline=bb_color, width=bb_line_width)

    #statistics = f"included_bbox/total_bbox_detected: {num_bbox_included}/{len(bboxes)}"
    #draw.text( (100,100), statistics, fill=(0,0,255,255))
    return pil_image_cp, num_bbox_included

#----------------------------------------
# update title of image with bb
#----------------------------------------
def update_bb_image_title( bb_image_ax, num_bb_selected, num_bb_total):
    title_text =f"Image with bb ({num_bb_selected}/{num_bb_total})\n" + \
            f"at bb_score_threshold {bb_score_threshold:.2f}"

    bb_image_ax.title.set_text(title_text)
    return



if __name__ == '__main__':

    # command line arguments definition
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--version', choices=['v1', 'v2'], default = 'v2',
                        help='set detection model version')
    parser.add_argument('--image',
                        required=True,
                        help='input image file path')
    parser.add_argument('--threshold', type=float, default = 0.5,
                        help='threshold for inclusion of bbox with its score.\n' +
                              'valid range: 0.0 <= threshold <= 1.0')

    # command line argument parsing
    args =  parser.parse_args()
    version = args.version
    image_file_path = args.image
    global bb_scorethreshold
    bb_score_threshold = args.threshold

    # set model conf file path and mode weight file path
    # prefixed by ./model/[detection|recognition]
    config_file = os.path.join('model/detection', detect_model[version]["config_file"])
    weight_file = os.path.join('model/detection', detect_model[version]["weight_file"])

    # print cfg file path and weight file path (DEBUG)
    print(f"config file path: {config_file}")
    print(f"weight file path: {weight_file}")

    # clone project level config and merge with experiment config
    cfg = cfg.clone()
    cfg.merge_from_file(config_file)

    # Detection model creation
    demo = DetectionDemo(cfg, weight_file)

    # open image file as PIL.Image
    pil_image = Image.open(image_file_path).convert('RGB')
    org_pil_image = np.array(pil_image)


    #----------------------------------------
    # bboxes inference on input image
    #----------------------------------------
    prediction = demo.run_on_pil_image(pil_image)

    bboxes_list = prediction['bboxes']
    scores_list = prediction['scores']

    num_bb_total = len(bboxes_list)

    #----------------------------------------
    # subplots for original image and image with bboxes
    #----------------------------------------
    # https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
    fig, axes = plt.subplots(1,2)


    # reserve space for slider, buttons
    # ref: https://pinkwink.kr/824
    plt.subplots_adjust(left=0.25, bottom=0.1)

    #---------------------------------
    # radio button for bbox width
    #---------------------------------
    rb_width_ax = plt.axes([0.025, 0.4, 0.1, 0.2], # axes(rect) -> rect: left, bottom, width, height
                     facecolor='lightgoldenrodyellow')

    rb_width = RadioButtons(rb_width_ax, (1, 2, 3, 4), active=1)

    # radio button callback function
    def update_bb_line_width(val):
        global bb_line_width

        new_bb_line_width = int(val)
        if bb_line_width == new_bb_line_width:
            return

        # set new bb_line_width, if changed
        bb_line_width = new_bb_line_width

        # redraw when bb_line_width change
        pil_image_cp, num_bb  = bb_image_draw(pil_image, bb_line_width=bb_line_width,
                                     bb_color=bb_color, bb_score_threshold=bb_score_threshold)
        axes[1].imshow(np.array(pil_image_cp))
        plt.draw()

    # conect radio button to its callback
    rb_width.on_clicked(update_bb_line_width)

    #---------------------------------
    # radio button for bbox color
    #---------------------------------
    rb_color_ax = plt.axes([0.025, 0.7, 0.1, 0.2], # axes(rect) -> rect: left, bottom, width, height
                     facecolor='lightgoldenrodyellow')

    rb_color = RadioButtons(rb_color_ax, ('red','green', 'blue', 'purple'), active=0)

    # radio button callback function
    def update_bb_color(val):
        global bb_color

        new_bb_color = val
        if bb_color == new_bb_color:
            return

        # set new bb_color, if changed
        bb_color = new_bb_color

        # redraw when bb_color change
        pil_image_cp, num_bb  = bb_image_draw(pil_image, bb_line_width=bb_line_width,
                                     bb_color=bb_color, bb_score_threshold=bb_score_threshold)
        axes[1].imshow(np.array(pil_image_cp))
        plt.draw()

    # conect radio button to its callback
    rb_color.on_clicked(update_bb_color)

    #---------------------------------
    # slider fo bbox score threshold
    #---------------------------------
    slider_ax = plt.axes([0.25, 0.05, 0.6, 0.03], # axes(rect) -> rect: left, bottom, width, height
                     facecolor='lightgoldenrodyellow')

    slider_threshold = Slider(slider_ax, 'bb_score_threshold:', 0.0, 1.0, 
                              valstep=0.05, valinit=bb_score_threshold)

    # slider callback function
    def update_bbox_score_threshold(val):
        global bb_score_threshold

        new_bb_score_threshold = float(val)

        if bb_score_threshold == new_bb_score_threshold:
            return

        # set new bb_score_threshold, if changed
        bb_score_threshold = new_bb_score_threshold

        # redraw when bb_line_width change
        pil_image_cp, num_bb  = bb_image_draw(pil_image, bb_line_width=bb_line_width,
                                     bb_color=bb_color, bb_score_threshold=bb_score_threshold)
        axes[1].imshow(np.array(pil_image_cp))

        update_bb_image_title( axes[1], num_bb, num_bb_total)

        plt.draw()


    # conect slider to its callback
    slider_threshold.on_changed(update_bbox_score_threshold)


    #---------------------------------
    # Initial image draw
    #---------------------------------
    pil_image_cp, num_bb = bb_image_draw(pil_image, bb_line_width=bb_line_width,
                                 bb_color=bb_color, bb_score_threshold=bb_score_threshold)
    axes[0].imshow(org_pil_image)
    axes[1].imshow(np.array(pil_image_cp))

    # display x-axis on the top
    axes[0].xaxis.tick_top()
    axes[1].xaxis.tick_top()

    # add minor tick into both axes
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0].tick_params(which='minor', length=5, colors='red')

    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1].tick_params(which='minor', length=5, colors='red')

    # set subplot background for contrast
    axes[0].set_facecolor('xkcd:salmon')
    axes[1].set_facecolor('xkcd:salmon')

    #axes[0].axis('off')
    #axes[1].axis('off')

    w, h = pil_image.size
    mode = pil_image.mode
    axes[0].title.set_text(f'Input Image\n{h}x{w} {mode}')
    update_bb_image_title( axes[1], num_bb, num_bb_total)


    # https://stackoverflow.com/questions/42435446/how-to-put-text-outside-python-plots
    # coord to place the text are in figure coord, 
    # where (0.0) is the bottom left and (1,1) is the top rigt of the figure
    sub_text =f"Detection Model Version:{version}\n" + \
            f"Prediction Time: {((end_time-start_time).microseconds)/1000.0:5.2f} ms."
    plt.gcf().text(0.5, 0.1, sub_text, fontsize=8, color='blue')

    plt.show()



