# Script for plotting images used for paper writing.
from os.path import join
from math import floor, ceil

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import savemat
from numpy import clip, quantile
from util.functions import figure_preproduction, axis_rectangle_labeling

RR = (8, 9, 10, 11)
MODEL_PATH = "..\\models"


class Figures:
    @staticmethod
    def monosample_recognitions_by2method(img, truths, predictions, recognitions, savefname=None,
                                          method_names=("truth", "prediction", "recognition"),
                                          colorstr=("green", "blue", "pink"), **kwargs):
        """ Plots the recognition results of neural network as well as automatic recognition. """
        fig, axs = figure_preproduction(1, 3, **kwargs)
        for ax, labels, mtd in zip(axs, (truths, predictions, recognitions), method_names):
            ax.set_title(mtd)
            ax.imshow(img, cmap="gray")
            axis_rectangle_labeling(ax, labels[:, :4], np.take(colorstr, labels[:, 4].astype(np.int8)))
        if savefname:  # save figures
            fig.savefig(savefname, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def accuracy_dl_vs_traditional(counter, avg_min_distances, savefname="", max_distance=25, savemat_name=""):
        """ Explore the recognition accuracy of Deeplearning and Traditional method on the variety RBC densities.
        Illustrates the data with a plots, and save the accuracy data with 'mat' file, for further plotting.
        """
        # finding the (image) samples that have the avg min distance less than threshold.
        distance_indexes, = np.where(avg_min_distances < max_distance)
        # gives the sorted indexes of samples through comparing the avg min distances.
        inv_order_indexes = sorted(distance_indexes, key=lambda x: avg_min_distances[x], reverse=True)
        # for each sample, gets the rbc counts both for dl predictions and traditional recognitions
        # Notice: the 0 index in the second axis for indexing v['statistics'] array is representing rbc data position.
        rbc_counter = {k: v["statistics"][:, 0][inv_order_indexes] for k, v in counter.items()}

        # Dividing to individual range boxes
        # mapping the average_mini_distance to real scale with micrometer metrics
        xlabels = 4.8 / 10 * avg_min_distances[inv_order_indexes]  # pixel_size*magnification*distance
        # generates the container for each bar according to the sample avg_min_distance
        bar_boxes = {f"{x}-{x + 1}": [0, 0, 0] for x in range(6, 12)}
        for xlb, tu_count, dl_count, td_count in zip(
                xlabels, rbc_counter["truths"], rbc_counter["predictions"], rbc_counter["recognitions"]):
            bot, top = floor(xlb), ceil(xlb)
            bar_boxes[f"{bot}-{top}"][0] += tu_count
            bar_boxes[f"{bot}-{top}"][1] += dl_count
            bar_boxes[f"{bot}-{top}"][2] += td_count

        dl_accuracy = [bar_boxes[f"{x}-{x + 1}"][1] / bar_boxes[f"{x}-{x + 1}"][0] for x in range(6, 12)]
        td_accuracy = [bar_boxes[f"{x}-{x + 1}"][2] / bar_boxes[f"{x}-{x + 1}"][0] for x in range(6, 12)]

        # Ploting
        fig, ax = figure_preproduction(figsize=(4, 4), ticks=True, spines=True)
        ax.plot(dl_accuracy)
        ax.plot(td_accuracy)
        ax.xaxis.set_inverted(True)
        if savefname:  # save figures
            fig.savefig(savefname, dpi=150)
        else:
            plt.show()
        if savemat_name:
            savemat(savemat_name, {"dl_accuracy": dl_accuracy, "td_accuracy": td_accuracy})

    @staticmethod
    def labels_accuracy_child_plot(image, labels, ax=None, xlabels=None, linewidth=None,
                                   save_fpath=None, dpi=150, figsize=None, fontsize=3, labelpad=.5):
        """ labels: {"coordinates": [n_cells, 4], "colors": [n_cells,]} """
        fontsize, labelpad = fontsize, labelpad  # points
        imh, imw = [x / dpi for x in image.shape]
        figh = imh + (fontsize + labelpad) / 72
        if figsize is None:
            figsize = (imw, figh)

        if not ax:
            _, ax = figure_preproduction(figsize=figsize, spines=False, ticks=False, constrained_layout=False)
        ax.imshow(image, cmap="gray")
        if xlabels:
            ax.set_xlabel(xlabels, labelpad=labelpad, fontdict={"fontsize": fontsize})
        for (x, y, w, h), c in zip(labels["coordinates"], labels["colors"]):
            rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=False, color=c, lw=linewidth)
            ax.add_patch(rect)
        if save_fpath:
            plt.tight_layout(pad=0)
            plt.savefig(save_fpath, dpi=dpi, pad_inches=0)
            plt.close()
        return ax

    @staticmethod
    def density_variety_sample_ids(filename):
        # read the names of file to be processed
        with open(filename, "r") as f:
            sample_fileids = [line.rstrip("\n") for line in f]
        return sample_fileids

    @staticmethod
    def export_recognized_figures(dataset, sample_file_ids, imsource_path="", export_fpath="",
                                  fig_dpi=300, box_colors=("green", "blue", "purple")):
        # For every sample, plots a figure
        for fileid in tqdm(sample_file_ids, desc="Exporting comparison"):
            sample_idx = dataset["source_fnames"].index(join(imsource_path, fileid + ".bmp"))
            labels = [(label_type, dataset.available_labels(label_type)[sample_idx])
                      for label_type in ("truths", "predictions", "recognitions")]
            phase_image = dataset["phase"][sample_idx]
            for labels_type, labels in labels:
                # construct input labels for bbox plotting
                coord_labels = {"coordinates": labels[:, :4], "colors": np.take(box_colors, labels[:, 4].astype(int))}
                # Adjust array contrast for more clear display
                phase_image = clip(phase_image, quantile(phase_image, 0.003), quantile(phase_image, 0.997))
                # The label string of the x axis on the plotted figure
                xlabelstr = "%s_%s %.2f" % (fileid, labels_type, dataset.set["average_mini_distance"][sample_idx])
                # If save, and where to save
                save_fname = join(export_fpath, "%s_%s.svg" % (fileid, labels_type))
                Figures.labels_accuracy_child_plot(image=phase_image, labels=coord_labels, xlabels=xlabelstr,
                                                   linewidth=0.4, save_fpath=save_fname, dpi=fig_dpi)
