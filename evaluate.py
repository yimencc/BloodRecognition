import pickle
from collections import OrderedDict
from os.path import exists, basename

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.pyplot import Rectangle

from Autodetect.annotation import Recognizer
from Deeplearning.util.losses import FeatureTranslator
from Deeplearning.util.dataset import DST_IMGSZ, GRIDSZ
from Deeplearning.bloodsmear import cross_minimal_eudistance
from Deeplearning.util.functions import (shape2fourpoints, figure_preproduction, source_from_sample_id,
                                         iou_assembling, boxes_area_merge)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Notice: scripts eval of Deeplearning should be able to evaluate model performance during model training
# and the output of executing function should output as the same formats as the labels.


def non_maximum_suppress(confidences, coordinates, over_thres=.4):
    """ Params
        TODO: wrapping the non_maximum_suppress and channel nms as an integral module of the neural network
        TODO: Considering is it possible to archive different overlap threshold on rbc cells and platelet.
        ------
        confidences:    {Tensor: (n_posi_m,)},
        coordinates:    {Tensor: (n_posi_m, x-y-w-h)},
        class_scores:   {Tensor: (n_posi_m, n_class)},
        over_thresh:    float
    """
    x1, y1, x2, y2 = coordinates.permute(1, 0)
    areas = (x2 - x1) * (y2 - y1)
    order = confidences.argsort()
    pick = []
    while len(order):
        pick.append(idx := order[-1])

        xx1 = torch.maximum(x1[order[:-1]], x1[idx])
        yy1 = torch.maximum(y1[order[:-1]], y1[idx])
        xx2 = torch.minimum(x2[order[:-1]], x2[idx])
        yy2 = torch.minimum(y2[order[:-1]], y2[idx])

        # inter: w * h
        inter = torch.clamp(xx2 - xx1, min=0.) * torch.clamp(yy2 - yy1, min=0.)
        over = inter / (areas[idx] + areas[order[:-1]] - inter)
        order = order[:-1][over < over_thres]
    return pick


class Prediction:
    def __init__(self, data, image_shape, grid_size, anchors, n_class, device, conf_threshold, nms_threshold):
        if data.device.type != device:
            data = data.to(device).detach()

        self.ncls = n_class
        self.device = device
        self.gdsz = grid_size
        self.nanc = len(anchors) // 2
        self.translator = FeatureTranslator(data, anchors, device, grid_size, False)

        # Data Operations
        positive, confidence = self.anchors_filtering(conf_threshold)
        self.positive = self.box_non_maximum_suppression(positive, confidence, nms_threshold)
        self.match_real_shape(image_shape)

    def anchors_filtering(self, conf_threshold):
        # data shape: (40, 40, 4, 8).  last dim: (x, y, w, h, conf, sc1, sc2, sc3)
        boxes, confidences = [], []
        for i in range(self.translator.data.shape[0]):
            for j in range(self.translator.data.shape[1]):
                # Chose the best confidence anchor box
                conf_idx = torch.argmax(self.translator.conf[i, j])  # conf (40, 40, 4) -> array([i])
                # Decide weather collect
                if self.translator.conf[i, j, conf_idx] > conf_threshold:
                    coord = torch.cat([self.translator.xy[i, j, conf_idx], self.translator.wh[i, j, conf_idx]], -1)
                    label = torch.argmax(self.translator.scores[i, j, conf_idx])
                    confidences.append(self.translator.conf[i, j, conf_idx])
                    boxes.append(torch.cat([coord, torch.tensor([label])]))

        positive, confidence = torch.stack(boxes), torch.stack(confidences)
        return positive, confidence

    def box_non_maximum_suppression(self, positive, confidence, nms_threshold):
        # for each sample, generate the picked indexes through compute NMS
        coordinates = shape2fourpoints(positive[:, :4], (self.gdsz, self.gdsz))
        picks = non_maximum_suppress(confidence, coordinates, nms_threshold)
        picks = torch.tensor(picks)
        return positive[picks]

    def match_real_shape(self, image_shape):
        """ Scale the output data to match the original image size. """
        scales = torch.tensor(image_shape[::-1], device=self.device) / self.gdsz
        # new_attrib_val: {list: 8} [(n_posi_m, n_anc, n_attrib), ...]
        self.positive[:, :2] = self.positive[:, :2] * scales
        self.positive[:, 2:4] = self.positive[:, 2:4] * scales


def dataset_prediction(dataloader, model, **prediction_params):
    """ Processing an dataset, and provide numpy data. """
    def available_truth(truth_tensor):
        truth = truth_tensor.numpy()  # (b, max_box_num, 5)
        return [sp_truth[np.mean(sp_truth[..., :4], axis=-1) > 0] for sp_truth in truth]

    def coordinate_transform(batch_coordinates, scale=DST_IMGSZ / GRIDSZ):
        for sp_coordinates in batch_coordinates:
            sp_coordinates[..., :4] = sp_coordinates[..., :4] * scale
        return batch_coordinates

    phase_fnames, predict_boxes, truths = [], [], []
    for modalities, label, (_, modality_names) in tqdm(dataloader, desc="Model Processing"):
        # Model FP prediction
        model_output = model(modalities.to("cuda")).to("cpu").detach_()
        # Processing model output
        for data in model_output:
            prediction = Prediction(data, **prediction_params)
            predict_boxes.append(prediction.positive.numpy())

        # Collects the truth data from test dataset
        truths_available = available_truth(label[3])
        truths += coordinate_transform(truths_available)
        phase_fnames += modality_names[1]

    return OrderedDict({"phase_fnames": np.array(phase_fnames),
                        "predict_boxes": np.array(predict_boxes, dtype=object),
                        "truths": np.array(truths, dtype=object)})


def render_batch_sample_predictions(positives, batch_sample, color=("g", "b", "purple")):
    images = batch_sample.to("cpu").detach() if batch_sample.device.type == "cuda" else batch_sample.detach()
    # config artist assign
    n_row, n_col, figsize = [2, 4, (15, 7.5)] if len(images) == 8 else [2, 2, (8, 8)]
    fig, axis = figure_preproduction(n_row, n_col, figsize)
    for i, (mods, positive) in enumerate(zip(images, positives)):
        ax = axis[divmod(i, n_col)]
        ax.imshow(mods[1], cmap="gray")
        for x, y, w, h, lbl in positive:
            ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False, color=color[int(lbl)]))
    plt.show()


def model_evaluation_visualize(dataloader, model, **prediction_params):
    """ Testing the trained network models."""
    for modalities, _ in dataloader:
        model_outputs = model(modalities.to("cuda")).to("cpu").detach_()
        predict_boxes = []
        for data in model_outputs:
            prediction = Prediction(data, **prediction_params)
            predict_boxes.append(prediction.positive.numpy())
        render_batch_sample_predictions(predict_boxes, modalities)


def merge_platelets_to_assembles(boxes, platelet_flag=2.):
    platelet_indexes, = np.where(boxes[:, 4] == platelet_flag)
    platelet_boxes = boxes[platelet_indexes]
    non_revised_boxes = np.delete(boxes, platelet_indexes, axis=0)
    assembled_boxes = iou_assembling(platelet_boxes)
    assembles = []
    for boxes in assembled_boxes:
        assembles.append(boxes_area_merge(boxes))
    return np.concatenate([non_revised_boxes, np.array(assembles)])


class RecognitionAnalyzer:
    """
    Packaging the recognition results of BSRNet and EdgeContour method, generating
    corresponding statistic data.
    TODO: Build this class as an universal tools class for both BSRNet and EdgeContour method.
    """
    def __init__(self, source_filename=None, refresh=False):
        self.data = {}
        self.src_filename = source_filename
        if self.src_filename is not None:
            if exists(self.src_filename) and not refresh:
                with open(self.src_filename, "rb") as f:
                    self.data = pickle.load(f)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.data[key] = value
        elif isinstance(key, int):
            for k, v in value.items():
                self.data[k][key] = v

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.data.get(item)
        elif isinstance(item, int):
            return {k: v[item] for k, v in self.data.items()}
        else:
            raise Exception()

    def dump_data(self):
        with open(self.src_filename, "wb") as f:
            pickle.dump(self.data, f)

    @staticmethod
    def available_indexes(sample_coordinates):
        """ sample_coordinates: (n_cells, 5) """
        indexes, = (np.mean(sample_coordinates[:, :4], axis=1) != 0).nonzero()
        return indexes

    def average_mini_distances(self, truths):
        average_mini_distance = []  # Adding an variable to describe cell density
        for truth in tqdm(truths, desc="Calculating avg mini distances"):  # truth: {ndarray: (n_box_max, x-y-w-h-lbl)}
            avg_mini_distance = np.mean(cross_minimal_eudistance(truth[self.available_indexes(truth), :2])[1])
            average_mini_distance.append(avg_mini_distance)
        return np.array(average_mini_distance)

    @staticmethod
    def _calc_accuracy(maps, bins=3, val_range=(0., 2.)):
        counter = OrderedDict()
        for key in maps.keys():
            sampel_labels = (sample_boxes[:, 4] for sample_boxes in maps[key])
            labels_counts = np.array([np.histogram(label, bins, val_range)[0] for label in sampel_labels])
            counter[key] = {"statistics": labels_counts,                # three types cell numbers in each sample
                            "summary": np.sum(labels_counts, axis=0)}   # Total numbers of three types cell
        return counter

    @property
    def counter(self):
        """ Calculate the accuracy of three method """
        return self._calc_accuracy({k: self[k] for k in ("Truths", "BSRNet", "EdgeContour")})

    def deep_learning_recognitions(self, model, dataloader, **pred_params):
        """ Blood Smear image recognition -> cell properties extraction -> cell statistic
        predictions: {dict:4} - {"phase_fnames": [], "coordinates": [], "labels": [], "truths": []} """
        # The pre_results contains the manual-labeled truth data and the predictions of neural network
        previous = dataset_prediction(dataloader, model, **pred_params)
        self.data.update({"Truths": previous["truths"], "BSRNet": previous["predict_boxes"]})
        # Calculates the average minimal distances of truth label samples
        self["avg_min_distance"] = self.average_mini_distances(self["Truths"])

        # Gathering Messages
        self.data.update({"phase": [], "source_fnames": []})
        for fname in tqdm(previous["phase_fnames"], desc="Gathering Image Messages"):
            phase = source_from_sample_id(basename(fname), "phase_matrix")
            self.data["phase"].append(phase)
            self.data["source_fnames"].append(fname)
        self["phase"] = np.array(self["phase"])

    def edge_contour_recognitions(self):
        # Auto-recognizing & relevant messages gathering
        self.data.update({"EdgeContour": []})
        for phase in tqdm(self.data["phase"], desc="Auto-recognizing Cells"):
            self.data["EdgeContour"].append(Recognizer.process(phase))
        # Integrates the list container to numpy array
        self["EdgeContour"] = np.array(self["EdgeContour"], object)
