import os
import sys
from os.path import join as oj
from functools import reduce, partial

import torch
import numpy as np
import matplotlib.pyplot as plt

from cccode import image
from evaluate import PredHandle
from models import YoloV4Model, load_model
from dataset import RbcDataset, DataLoader, ANCHORS, VALID_DS_CONSTRUCTOR

nx = np.newaxis
ck = image.Check()
MODEL_PATH = "..\\data\\models"


def sample_analyse(sample, model, thresh):
    """
    """
    inputs, labels = sample["image"], sample["label"]
    phand = PredHandle(inputs, model(inputs), ANCHORS)      # inputs [1, 1, 320, 320]

    def result_generator(input_tensor):
        for i, input_img_tensor in enumerate(input_tensor):
            input_img_numpy = input_img_tensor[0].numpy()
            label_true = [lb.numpy() for lb in labels[3][i] if any(lb)]
            label_pred = phand.real_coord_results(i, thresh)
            yield input_img_numpy, label_true, label_pred

    # fig_inputs = result_generator(inputs)
    # pred_show(fig_inputs, fig_path="..\\data\\models\\results", time_stamp="", idx=0, display=True)

    fig_inputs = result_generator(inputs)
    n_label, n_preds = [sum(elm) for elm in zip(*[(len(label), len(preds)) for _, label, preds in fig_inputs])]
    return n_label, n_preds


def yolo_evaluate():
    thresh = .7
    yolo_model = load_model()

    rbcDataset = RbcDataset(**VALID_DS_CONSTRUCTOR)
    valid_dataloader = DataLoader(rbcDataset, batch_size=8, shuffle=False)

    sample_results_gen = (sample_analyse(sp, yolo_model, thresh) for sp in valid_dataloader)
    total_label, total_preds = reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]), sample_results_gen)
    acc = 100*(total_preds-total_label)/total_label
    print(total_label, total_preds, acc)


def pred_show(inputs: list[list], fig_path: str, time_stamp: str,
              idx: int, display: bool, save_fig=False):
    """
    Evaluate the model accuracy, this func will plot several figures,
    each figure contain eight rbc recognition results, both for
    artifact annotation and deep learning annotation.
    """
    scale = 320/40
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    fig, axs = plt.subplots(2, 4, constrained_layout=True)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    total_n_label, total_n_preds = 0, 0
    for i, (input_img, bboxes, coord_2d) in enumerate(inputs):
        row, col = i // 4, i % 4
        ax = axs[row, col]
        ax.imshow(input_img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        coord_list = coord_2d[np.where(coord_2d.sum(-1) > 0)]
        for bbox in bboxes:
            x, y, w, h = [elm*320/40 for elm in bbox[:4]]
            rect = plt.Rectangle((x-w/2, y-h/2), width=w, height=h, lw=1.1,
                                 fill=False, color="blue")
            ax.add_patch(rect)
        for coord in coord_list:
            x1, y1, x2, y2 = [elm*scale for elm in coord]
            rect = plt.Rectangle((x1, y1), width=(x2-x1), height=(y2-y1), lw=1.1, fill=False, color="red", alpha=.8)
            ax.add_patch(rect)
        n_label = len(bboxes)
        n_pred = len(coord_list)
        total_n_label += n_label
        total_n_preds += n_pred
        ax.set_xlabel(f"Label: {n_label} Preds: {n_pred}, Error: "
                      f"{abs(n_pred-n_label)/n_label*100:.2f}%", fontsize=8)
    fig.suptitle(f"Total Error: {abs(total_n_preds-total_n_label)/total_n_label*100:.2f}%", fontsize=12)
    if save_fig:
        plt.savefig(os.path.join(fig_path, "%s_%d.png" % (time_stamp, idx)))
    if display:
        plt.show()


class Figure:
    """
    Results image display for annotation process
    """
    def __init__(self):
        self.dpi = 150
        self.crop_rate = (.3/.95, .3/.95)
        self.crop_fn = partial(image.View.crop, crop_rate=self.crop_rate)
        self.fig = None
        self.ax = None

    def figure_params(self, image_shape, constrained=True):
        pad_ratio = 0.2
        pxl_num_row, pxl_num_collum = image_shape
        fig_height, fig_width = ((1 + pad_ratio) * pxl_num_row / self.dpi,
                                 (1 + pad_ratio) * pxl_num_collum / self.dpi)
        constrained_layout = True if constrained else False
        fig_params = {"figsize": (fig_height, fig_width),
                      "constrained_layout": constrained_layout}
        return fig_params

    @staticmethod
    def indicators_setting(ax, ticks=False, spines=False):
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if not spines:
            for sp in ["top", "bottom", "right", "left"]:
                ax.spines[sp].set_visible(False)
        return ax

    def single_image_figure(self, img, fig_name="", cmap="gray", vmin=None, vmax=None, colorbar=False, show=False):
        # ck.hist(img)
        fig_params = self.figure_params(img.shape)
        self.fig, self.ax = plt.subplots(**fig_params)
        mp = self.ax.imshow(img, cmap=cmap, interpolation="antialiased", vmin=vmin, vmax=vmax)

        self.ax = self.indicators_setting(self.ax)
        if colorbar:
            plt.colorbar(mp, ax=self.ax, fraction=.1)

        # Save or Print
        if fig_name:
            plt.savefig(fig_name, dpi=self.dpi)
        if show:
            plt.show()

    def geometry_plot(self, bg_image, circle_locations, implement="show", **fig_params):
        fig, ax = plt.subplots(constrained_layout=True, **fig_params)
        ax.imshow(bg_image, cmap="gray")
        self.indicators_setting(ax)
        scale = 320/40
        for location in circle_locations:
            # location: (y, x, r)
            x1, y1, x2, y2 = [elm*scale for elm in location]
            rect = plt.Rectangle((x1, y1), width=(x2-x1), height=(y2-y1), lw=1.1, fill=False, color="red", alpha=.8)
            ax.add_patch(rect)
        if implement == "show":
            plt.show()
        else:
            return fig, ax


class PaperFigure3:
    def __init__(self, batch_size=8):
        self.thresh = .7
        self.batch_size = batch_size
        plan_name = "plan_5.3"
        model_name = "yolov2_0506-195652.pth"

        self.model = YoloV4Model()
        cur_model_fname = oj(MODEL_PATH, plan_name, model_name)
        self.model.load_state_dict(torch.load(cur_model_fname))

        rbcDataset = RbcDataset(**VALID_DS_CONSTRUCTOR)
        self.valid_dataloader = DataLoader(rbcDataset,
                                           batch_size=batch_size,
                                           shuffle=False)

    def network_output_conf_map(self):
        input_image, output_conf_map, label_pred = None, None, None
        for i, sample in enumerate(self.valid_dataloader):
            if i == 7:
                inputs, labels = sample["image"], sample["label"]
                # inputs [1, 1, 320, 320]
                phand = PredHandle(self.model(inputs), ANCHORS)
                phand.decompose()

                input_image = inputs.to("cpu").numpy()[0, 0]
                output_conf_map = phand.conf[0]

                phand.nms_along_box()
                label_pred = phand.real_coord_results(0, self.thresh)

        output_conf_map = output_conf_map.squeeze().sum(axis=(-1))

        fig = Figure()
        for i, img in enumerate([input_image, output_conf_map]):
            if i == 1:
                vmin = 0
                vmax = 1
            else:
                vmin = None
                vmax = None
            fig.single_image_figure(img, show=True, vmin=vmin, vmax=vmax)
        fig.geometry_plot(input_image, label_pred)

    def pred_results(self):
        input_image, output_conf_map, label_pred = None, None, None
        for i, sample in enumerate(self.valid_dataloader):
            inputs, labels = sample["image"], sample["label"]
            # inputs [1, 1, 320, 320]
            phand = PredHandle(self.model(inputs), ANCHORS)
            phand.decompose()

            for n in range(self.batch_size):
                input_image = inputs.to("cpu").numpy()[n, 0]
                output_conf_map = phand.conf[0]

                phand.nms_along_box()
                label_pred = phand.real_coord_results(0, self.thresh)


def figure_plot():
    fig3 = PaperFigure3()
    fig3.network_output_conf_map()


if __name__ == "__main__":
    """Run Entrance"""
    print(sys.version_info, "\n", sys.version)
    figure_plot()
