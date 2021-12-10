import pickle
import logging
from logging import config
from os.path import exists, basename, join
from collections import OrderedDict, namedtuple

import torch
import numpy as np
from tqdm import tqdm
from torch import tensor
from matplotlib import pyplot as plt
from matplotlib.pyplot import Rectangle
from numpy import concatenate as concate

from Autodetect.annotation import Recognizer
from Deeplearning.util.losses import FeatureTranslator
from Deeplearning.util.functions import shape2fourPoints, figure_preproduction
from Deeplearning.bloodsmear import (outlier_rbc_analysis, find_source_from_sample_id,
                                     cross_minimal_euclidean_distance)
from Deeplearning.util.dataset import (create_dataloader, SET202109_1_FILENAME, NameRememberedDataset,
                                       DST_IMGSZ, GRIDSZ, ANCHORS, N_CLASSES)

WAVELENGTH = 632.8e-9
MODEL_PATH = "..\\models"
INVES_ROOT = "D:\\Postgraduate\\Investigate\\RBC Recognition"

# Notice: scripts eval of Deeplearning should be able to evaluate model performance during model training
# and the output of executing function should output as the same formats as the labels.
# TODO: Integrating the NMS into an torch model


def non_maximum_suppress(confidences, coordinates, image_shape, over_thres=.4):
    """ Params
        TODO: wrapping the non_maximum_suppress and channel nms as an integral module of the neural network
        TODO: Considering is it possible to archive different overlap threshold on rbc cells and platelet.
        ------
        confidences:    {Tensor: (n_posi_m,)},
        coordinates:    {Tensor: (n_posi_m, x-y-w-h)},
        class_scores:   {Tensor: (n_posi_m, n_class)},
        over_thresh:    float
    """
    x1, y1, x2, y2 = shape2fourPoints(coordinates, image_shape).permute(1, 0)
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


class PredHandle:
    """
    Attributes
    ----------
    preds: FeatureTranslator
        Container used to organize model outputs
    positive_grids: OrderedDict
        Stored the processed model predictions of a batch sample.
        positive_grids:  ('xy'      {list: 8}    [(N_posi_0, N_anc, x-y),     ...]
                          'wh'      {list: 8}    [(N_posi_0, N_anc, w-h),     ...]
                          'conf'    {list: 8}    [(N_posi_0, N_anc),          ...]
                          'scores'  {list: 8}    [(N_posi_0, N_anc, N_class), ...]
    """
    def __init__(self, data, image_shape, grid_size, anchors, n_class, device, conf_thres, nms_overlapping_thres):
        """ Predicting the red blood cell inside given images.\n
        Parameters
        -----------
        data: torch.Tensor
            the outputs of model FP
        image_shape: tuple
        grid_size: int
        anchors: list
        n_class: int
        device: str
        conf_thres: float
        nms_overlapping_thres: float
        """
        if data.device.type != device:
            data = data.to(device).detach()
        self.positive_grids = OrderedDict()
        self.preds = FeatureTranslator(data, grid_size, anchors, n_class, device)

        # Data Operations
        self.confidence_filter(conf_thres)
        self.grid_non_maximum_suppression(over_thres=nms_overlapping_thres)
        self.match_real_shape(image_shape)

    def __len__(self):
        return len(self.preds.data)

    def __getitem__(self, item):
        """ return an tuple of sample prediction that have the attributes:
            pred - model output for corresponding sample
            xy - coordinates of predicted objects inside the sample
            wh - shape of predicted objects
            conf - confidence for object appearance
            labels - the object labels
        """
        if isinstance(item, int):
            nmd_tuple = namedtuple("Sample", ("pred", "xy", "wh", "conf", "labels"))
            data = (self.preds.data[item], *[self.positive_grids[at][item] for at in ("xy", "wh", "conf", "labels")])
            return nmd_tuple(*data)
        else:
            assert item in ("xy", "wh", "conf", "labels")
            val = self.positive_grids.get(item)
            val = [elm.to("cpu").detach() if elm.device.type == "cuda" else elm for elm in val]
            return val

    def numpy_get(self, item):
        if isinstance(item, str):
            val = self[item]
        else:
            raise KeyError
        return [elm.numpy() for elm in val]

    def confidence_filter(self, threshold, attributes=("xy", "wh", "conf", "scores")):
        """ Screen out the grids that is likely to have object in it,
        this boxes own confidence better than threshold. """
        # TODO: use adaptive confidence for cell filtering, e.g. RBC: 0.7, WBC: 0.5, Platelet: 0.6
        bbox_bestconf = self.preds.conf.amax(-1)  # (b, gdsz, gdsz)
        # positive_indexes: {list: 8} [(2, n_posi0), (2, n_posi1), ..., (2, n_posi7)]
        positive_indexes = [torch.nonzero(tensor(conf_slc >= threshold), as_tuple=True) for conf_slc in bbox_bestconf]

        for attrib in attributes:
            # (b, gdsz, gdsz, n_anc, n_attrib)
            attrib_value = getattr(self.preds, attrib)
            # attrib_slc: (gdsz, gdsz, n_anc, n_prop), idx_slc: (2, n_posi_m) -> slice_value: (n_posi_m, n_anc, n_prop)
            slice_value = [attrib_slc[idx_slc] for attrib_slc, idx_slc in zip(attrib_value, positive_indexes)]
            self.positive_grids.update({attrib: slice_value})

        # Best anchors in each grid.
        self.anchors_campaign()

    def anchors_campaign(self):
        """ Select the Most confidence bbox from the given anchors candidates in each grid unit. (1 / N_anc) """
        if not self.positive_grids:
            raise Exception("positive_grids is empty, plz execute the confidence_filter method first.")
        # Prepare selection mask through founding the best confidence
        conf_oh_masks = []  # onehot_mask {list(bool): 8} [(n_posi_m, n_anc), ...]
        for conf in self.positive_grids.get("conf"):  # conf {Tensor: (n_posi_m, n_anc(4))}
            conf_argmax = torch.argmax(conf, dim=-1)  # (n_posi_m,)
            conf_mask = torch.stack([conf_argmax == ch for ch in range(self.preds.n_anchor)], -1)
            conf_oh_masks.append(conf_mask)

        # Selecting the bbox that have the best confidence in a grid.
        for attrib in self.positive_grids.keys():
            # attrib_val: {list: 8} [(n_posi_m, n_anc, n_attrib), ...]
            new_attrib_val = [attr_val[mask] for attr_val, mask in zip(self.positive_grids.get(attrib), conf_oh_masks)]
            self.positive_grids.update({attrib: new_attrib_val})  # update attributes

    def grid_non_maximum_suppression(self, **kwargs):
        """ Non-maximum suppression based-on IOU overlap threshold.
        Notice:This method is only works on the threshold filtered boxes, that means
        the prerequisite for this method is performing the 'confidence_filter' """
        # for each sample, generate the picked indexes through compute NMS
        pick_indexes = []
        gdsz = self.preds.grid_size
        properties = ("xy", "wh", "conf", "scores")
        for xy, wh, conf, scores in zip(*[self.positive_grids.get(item) for item in properties]):
            pick = non_maximum_suppress(conf, torch.cat([xy, wh], dim=-1), (gdsz, gdsz), **kwargs)
            pick_indexes.append(torch.tensor(pick, device=self.preds.device))

        # Update the positive grids with pick indexes generated from NMS
        for attr in self.positive_grids.keys():
            # attrib_val: {list: 8} [(n_posi_m, n_anc, n_attrib), ...]
            new_attrib_val = [attr_val[pick] for attr_val, pick in zip(self.positive_grids.get(attr), pick_indexes)]
            self.positive_grids.update({attr: new_attrib_val})

        # Adding an element 'class' representing the predicted object class to the positive_grids
        # According to the 'scores' for each object class, the object could be determined by the best one.
        obj_classes = [torch.argmax(score, dim=-1) for score in self.positive_grids["scores"]]
        self.positive_grids.update({"labels": obj_classes})

    def match_real_shape(self, image_shape):
        """ Scale the output data to match the original image size. """
        scales = torch.tensor(image_shape, device=self.preds.device) / self.preds.grid_size
        for attrib in ("xy", "wh"):
            # new_attrib_val: {list: 8} [(n_posi_m, n_anc, n_attrib), ...]
            new_attrib_val = [attrib_val * scales for attrib_val in self.positive_grids.get(attrib)]
            self.positive_grids.update({attrib: new_attrib_val})  # update attributes

    def predict_collection_show(self, images, focus="phase"):
        images = images.to("cpu").detach() if images.device.type == "cuda" else images.detach()
        properties = [self[item] for item in ("xy", "wh", "conf", "labels")]
        # config artist assign
        n_row, n_col, figsize = [2, 4, (15, 7.5)] if len(images) == 8 else [2, 2, (8, 8)]
        fig, axis = figure_preproduction(n_row, n_col, figsize)
        for i, (mods, xy, wh, conf, labels) in enumerate(zip(images, *properties)):
            ax = axis[divmod(i, n_col)]
            ax.imshow(mods[1] if focus == "phase" else None, cmap="gray")
            for (x, y), (w, h), cf, lbl in zip(xy, wh, conf, labels):
                ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, fill=False, color=("green", "blue", "purple")[lbl]))
        plt.show()

    @classmethod
    def iterate_dataset(cls, model, dataloader: NameRememberedDataset, **pred_params):
        """ Processing an dataset, and provide numpy data. """
        phase_fnames, coordinates, pred_labels, truths = [], [], [], []
        for modalities, label, (_, modality_names) in tqdm(dataloader, desc="Model Processing"):
            # Model FP prediction
            model_output = model(modalities.to("cuda")).to("cpu").detach_()
            # Processing model output
            phand = cls(model_output, **pred_params)
            truths.append(label[3].numpy())
            phase_fnames += modality_names[1]
            pred_labels += phand.numpy_get("labels")
            coordinates += [concate([xy, wh], -1) for xy, wh in zip(phand.numpy_get("xy"), phand.numpy_get("wh"))]

        predictions = OrderedDict({"phase_fnames": np.array(phase_fnames),
                                   "labels": np.array(pred_labels, dtype="object"),
                                   "coordinates": np.array("coordinates", dtype=object),
                                   "truths": concate(truths, 0)})
        return predictions


def model_evaluation_visualize(dataloader, model_fpath, pred_params):
    """ Testing the trained network models."""
    logger.info("Loading Model")
    yolo_model = torch.load(model_fpath).to("cuda")
    for modalities, _ in tqdm(dataloader):
        # Processing model output
        pred_handle = PredHandle(yolo_model(modalities.to("cuda")).to("cpu").detach_(), **pred_params)
        pred_handle.predict_collection_show(modalities)


def rbc_properties_from_dl_for_set202109_1(predictions, cache_filename=""):
    """ RBC cells properties Analyse based on the prediction results of deep learning for SET202109-1 """
    rbc_properties = {}
    for fname, coords, labels in zip(*[predictions.get(k) for k in ("source_fnames", "coordinates", "labels")]):
        phase = find_source_from_sample_id(basename(fname), "phase_matrix")
        prop_dict = outlier_rbc_analysis(phase, coords, labels)
        rbc_properties.update(prop_dict if not rbc_properties else
                              {k: np.append(v, prop_dict.get(k)) for k, v in rbc_properties.items()})
    # Caching
    if cache_filename:
        with open(cache_filename, "wb") as f:
            pickle.dump(rbc_properties, f)
    return rbc_properties


class Set2021091Metadata:
    def __init__(self, source_filename=None, refresh=False):
        self.src_filename = source_filename
        if exists(source_filename) and not refresh:
            with open(source_filename, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = {}
        self._available_truth_indexes = None

    def get(self, item):
        return self.data.get(item)

    def gets(self, items):
        return [self.get(it) for it in items]

    @property
    def rbc_properties(self):
        return rbc_properties_from_dl_for_set202109_1(self.data)

    def dump_data(self):
        # Organizes the data as np.array
        for k, v in self.data.items():
            if isinstance(v, list):
                if isinstance(v[0], np.ndarray):
                    if len({elm.shape for elm in v}) != 1:
                        self.data[k] = np.array(v, dtype=object)
                    else:
                        self.data[k] = np.array(v)
                else:
                    self.data[k] = np.array(v)

        with open(self.src_filename, "wb") as f:
            pickle.dump(self.data, f)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.data.get(item)
        else:
            raise Exception("Not build yet!")

    def appends(self, kvmaps):
        for k, v in kvmaps.items():
            self.data[k].append(v)

    def available_labels(self, label_type):
        return np.array([coord[self.available_indexes(coord)] for coord in self[label_type]], dtype=object)

    @staticmethod
    def available_indexes(sample_coordinates):
        """ sample_coordinates: (n_cells, 5) """
        indexes, = (np.mean(sample_coordinates[:, :4], axis=1) != 0).nonzero()
        return indexes

    def average_mini_distances(self, dump=True):
        average_mini_distance = []  # Adding an variable to describe cell density
        for truth in tqdm(self.get("truths"), desc="Calculating avg mini distances"):  # estimate average mini_distances
            # truth:  {ndarray: (n_box_max, x-y-w-h-lbl)}
            avg_mini_distance = np.mean(cross_minimal_euclidean_distance(truth[self.available_indexes(truth), :2])[1])
            average_mini_distance.append(avg_mini_distance)
        self["average_mini_distance"] = np.array(average_mini_distance)
        if dump:
            self.dump_data()

    def calc_accuracy(self):
        self.data["counter"] = OrderedDict({"truths": {}, "predictions": {}, "recognitions": {}})
        for col in self.data["counter"].keys():
            available_indexes = np.array([self.available_indexes(coord) for coord in self[col]], dtype=object)
            available_labels = [coord[indexes][:, 4] for coord, indexes in zip(self[col], available_indexes)]
            labels_for_every_samples = np.array([np.histogram(label, 3, (0., 2.))[0] for label in available_labels])
            self["counter"][col] = {
                "statistics": labels_for_every_samples,  # The numbers of three types cell in each sample
                "summary": np.sum(labels_for_every_samples, axis=0)  # Total numbers of three types cell
            }

    @staticmethod
    def dataset_predict(model_fpath, pred_params):
        """ Dataset predictions for SET202109_1. """
        logger.info(f"Loading Model: {model_fpath} and Proceeding data ...")
        yolo_model = torch.load(model_fpath).to("cuda")
        dataloader = create_dataloader(SET202109_1_FILENAME, "test", 8, dataset_obj=NameRememberedDataset)
        predictions = PredHandle.iterate_dataset(yolo_model, dataloader, **pred_params)
        return predictions

    def auto_verse_dl(self, model_fpath, pred_params):
        """ Blood Smear image recognition -> cell properties extraction -> cell statistic
        predictions: {dict:4} - {"phase_fnames": [], "coordinates": [], "labels": [], "truths": []} """
        # The pre_results contains the manual-labeled truth data and the predictions of neural network
        previous = self.dataset_predict(model_fpath, pred_params)
        # and scale the truth label to sample coordinates
        self["truths"] = previous["truths"] * np.tile([8, 8, 8, 8, 1], (*previous["truths"].shape[:2], 1))
        # Calculates the average minimal distances of truth label samples
        self.average_mini_distances(dump=False)
        # and the prediction of nn should be re-formatted to suitable data structure: np.ndarray
        self["predictions"] = [np.concatenate([coord, label[:, np.newaxis]], 1) for coord, label
                               in zip(previous["coordinates"], previous["labels"])]

        # Auto-recognizing & relevant messages gathering
        self.data.update({"recognitions": [], "phase": [], "source_fnames": []})
        for fname in tqdm(previous["phase_fnames"], desc="Auto-recognizing Cells"):
            phase = find_source_from_sample_id(basename(fname), "phase_matrix")
            self.appends({"phase": phase, "source_fnames": fname, "recognitions": Recognizer.process(phase)})

        self.calc_accuracy()
        self.dump_data()


if __name__ == '__main__':
    # ================== Python interpreter initializing =================
    if not plt.get_backend() == "tkagg":
        plt.switch_backend("tkagg")
    # Script Constants
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging Config ---------------------------------------------------
    logging.config.fileConfig(".\\log\\config\\evaluate.conf")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # ========================== Scripts Execute =========================
    try:
        accSrcFilename = ".\\caches\\acc_compare_set202109-1.pkl"
        set2021091 = Set2021091Metadata(source_filename=accSrcFilename, refresh=True)

        confThres = 0.6
        overThres = 0.3
        pipeDevice = "cpu"
        modelName = join(MODEL_PATH, "plan_8.2\\yolov5_1109-163812.pth")
        predParams = {"image_shape": (DST_IMGSZ, DST_IMGSZ),
                      "grid_size": GRIDSZ, "anchors": ANCHORS, "n_class": N_CLASSES,
                      "device": pipeDevice, "nms_overlapping_thres": overThres, "conf_thres": confThres}
        set2021091.auto_verse_dl(modelName, predParams)

    except Exception as e:
        logger.exception(e)
