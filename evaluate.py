import os
import pickle
import logging
from copy import deepcopy
from os.path import join
from logging import config
from collections import OrderedDict, namedtuple

import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Rectangle
from skimage.measure import regionprops_table, perimeter
from skimage.segmentation import expand_labels
from skimage.filters import threshold_otsu, threshold_multiotsu

from Deeplearning.util.losses import Prediction
from Deeplearning.util.functions import shape2fourPoints, figure_preproduction
from Deeplearning.util.dataset import (create_dataloader, ANCHORS, GRIDSZ, N_CLASSES, BloodSmearDataset,
                                       DATA_ROOT, DST_IMGSZ, SET202109_FILENAME, get_original_modality_from_id)

WAVELENGTH  =   632.8e-9
MODEL_PATH  =   "..\\models"
CELL_MAPS   =   ("rbc", "wbc", "platelet")
# TODO: Accuracy Statistic, Extract Characters


def non_maximum_suppress(confidences, coordinates, image_shape, over_thres=.4):
    """ Params
        ------
        confidences:    {Tensor: (n_cred_m,)},
        coordinates:    {Tensor: (n_cred_m, x-y-w-h)},
        class_scores:   {Tensor: (n_cred_m, n_class)},
        over_thresh:    float
    """
    x1, y1, x2, y2 = shape2fourPoints(coordinates, image_shape).permute(1, 0)
    areas = (x2-x1) * (y2-y1)
    order = confidences.argsort()

    pick = []
    while len(order):
        pick.append(idx := order[-1])

        xx1 = torch.maximum(x1[order[:-1]], x1[idx])
        yy1 = torch.maximum(y1[order[:-1]], y1[idx])
        xx2 = torch.minimum(x2[order[:-1]], x2[idx])
        yy2 = torch.minimum(y2[order[:-1]], y2[idx])

        # inter: w * h
        inter = torch.clamp(xx2-xx1, min=0.) * torch.clamp(yy2-yy1, min=0.)
        over  = inter / (areas[idx] + areas[order[:-1]] - inter)
        order = order[:-1][over < over_thres]
    return pick


def area_extract(img, coordinates, otsu_bins=20, expand_distance=5):
    h_upper, w_upper = img.shape
    labeled_area = torch.zeros_like(img)
    # Extracting the available area from the give boxes
    for lbl, (x, y, w, h) in enumerate(coordinates):
        # Decide weather all the area inside the box is available
        box_xrange = torch.clamp(x-w/2, 0).int(), torch.clamp(x+w/2, max=w_upper).int()
        box_yrange = torch.clamp(y-h/2, 0).int(), torch.clamp(y+h/2, max=h_upper).int()
        x_start, y_start = box_xrange[0], box_yrange[0]

        # Extract available areas from boxes
        strict_area = img[slice(*box_yrange), slice(*box_xrange)]
        otsu_thres = threshold_otsu(strict_area.numpy(), nbins=otsu_bins)

        # Labeling available areas
        y_index, x_index = (strict_area >= otsu_thres).nonzero(as_tuple=True)
        labeled_area[[idx+y_start for idx in y_index], [idx+x_start for idx in x_index]] = lbl + 1
        # ck.img_show(strict_area, figsize=(3, 3), spines=False, ticks=False, colorbar=False)

    # Expanding the extracted areas
    expanded_areas = expand_labels(labeled_area, distance=expand_distance)

    # Re-thresholding of the expanded areas using Otsu-threshold
    foreground_labels = np.zeros_like(img)
    background_labels = np.zeros_like(img)
    for lbl in range(1, np.max(expanded_areas).astype(int)):
        focus_area = np.where(expanded_areas == lbl, img, 0)
        regions = np.digitize(focus_area, bins=threshold_multiotsu(focus_area))
        background_labels[regions == 1] = lbl
        foreground_labels[regions == 2] = lbl
    return foreground_labels, background_labels


def outlier_cells_visual(img, outlier_cells, outlier_minimals, figsize=(7, 7)):
    fig, ax = figure_preproduction(figsize)
    ax.imshow(img, cmap="gray")
    colors = ("green", "blue", "purple")
    for cell, mini in zip(outlier_cells, outlier_minimals):
        (x, y, w, h, lbl), (_io, _ip, dist) = cell, mini
        ax.add_patch(plt.Rectangle((x-w / 2, y-h / 2), w, h, fill=False, color=colors[lbl]))
        ax.annotate(f"{dist.item():.1f}", xy=(x-5, y+3), xycoords="data", size=8, color="red")
    plt.show()


def euclidean_distance_filtering(coordinates):
    """ Select outlier cells through Computing Euclidean distance.
    Outputs
    -------
    minimals: list
        [(original_cell_id, pointed_cell_id, distance)]
    """
    x, y = torch.tensor(coordinates).permute(1, 0)
    xx1, xx2 = torch.meshgrid(x, x)
    yy1, yy2 = torch.meshgrid(y, y)
    distances = torch.sqrt((xx1 - xx2) ** 2 + (yy1 - yy2) ** 2)
    distances += torch.eye(len(distances)) * torch.max(distances)   # Maximal the distance with itself

    minimal_distances, minimal_indexes = torch.min(distances, dim=-1)
    minimals = list(zip(torch.arange(len(coordinates)), minimal_indexes, minimal_distances))
    return minimals


def single_sample_outlier_rbc(phase, xy, wh, labels, distance_thres, otsu_bins):
    def _get_tuple(name, attribs=("x", "y", "w", "h", "label")):
        return namedtuple(name, field_names=attribs)

    # i_sp: index of sample, each slice of image, area_dict: container for cells
    area_dict = OrderedDict({"rbc": [], "wbc": [], "platelet": []})
    for _xy, _wh, lbl in zip(xy, wh, labels):   # Sorting
        sp_name = CELL_MAPS[lbl]
        area_dict[sp_name].append(_get_tuple(sp_name)(*_xy, *_wh, lbl))

    # Compute the distances with closest recognized object for each cell.
    all_cells: list = area_dict["rbc"] + area_dict["wbc"] + area_dict["platelet"]
    cells_minimals = euclidean_distance_filtering([(c.x, c.y) for c in all_cells])
    # Select the cells away from other more than the distance threshold.
    outlier_minimals = list(filter(lambda elm: elm[2] > distance_thres, cells_minimals))
    outlier_cells = [all_cells[mini[0]] for mini in outlier_minimals if mini[0] < len(area_dict["rbc"])]
    # outlier_cells_visual(phase, outlier_cells, outlier_minimals)

    # Area extract: Extract rbc cell area through the aided of phase image
    outlier_coordinates = [(c.x, c.y, c.w, c.h) for c in outlier_cells]
    foreground_labels, background_labels = area_extract(phase, outlier_coordinates, otsu_bins)
    return foreground_labels, background_labels


class PredHandle:
    """
    Attributes
    ----------
    preds: Prediction
        Container used to organize model outputs
    credible_grids: OrderedDict
        Stored the processed model predictions of a batch sample.
        credible_grids:  ('xy'      {list: 8}    [(N_cred_0, N_anc, x-y),     ...]
                          'wh'      {list: 8}    [(N_cred_0, N_anc, w-h),     ...]
                          'conf'    {list: 8}    [(N_cred_0, N_anc),          ...]
                          'scores'  {list: 8}    [(N_cred_0, N_anc, N_class), ...]
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
        self.credible_grids = OrderedDict()
        self.preds = Prediction(data, grid_size, anchors, n_class, device)

        # Data Operations
        self.confidence_filter(conf_thres)
        self.grid_non_maximum_suppression(over_thres=nms_overlapping_thres)
        self.match_real_shape(image_shape)

    def get(self, item, to_cpu=True):
        assert item in ("xy", "wh", "conf", "labels")
        val = self.credible_grids.get(item)
        if to_cpu:
            val = [elm.to("cpu").detach() if elm.device.type == "cuda" else elm for elm in val]
        return val

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
        nmd_tuple = namedtuple("Sample", ("pred", "xy", "wh", "conf", "labels"))
        data = (self.preds.data[item], *[self.get(at)[item] for at in ("xy", "wh", "conf", "labels")])
        return nmd_tuple(*data)

    def confidence_filter(self, threshold, attributes=("xy", "wh", "conf", "scores")):
        """ Screen out the grids that is likely to have object in it,
        this boxes own confidence better than threshold.
        """
        # (b, gdsz, gdsz)
        bbox_best_confidences = self.preds.conf.amax(-1)
        # credible_indexes: {list: 8} [(2, n_cred0), (2, n_cred1), ..., (2, n_cred7)]
        credible_indexes = [(conf_slc >= threshold).nonzero(as_tuple=True) for conf_slc in bbox_best_confidences]

        for attrib in attributes:
            # (b, gdsz, gdsz, n_anc, n_attrib)
            attrib_value = getattr(self.preds, attrib)
            # attrib_slc: (gdsz, gdsz, n_anc, n_prop), idx_slc: (2, n_cred_m) -> slice_value: (n_cred_m, n_anc, n_prop)
            slice_value = [attrib_slc[idx_slc] for attrib_slc, idx_slc in zip(attrib_value, credible_indexes)]
            self.credible_grids.update({attrib: slice_value})

        # Best anchors in each grid.
        self.anchors_campaign()

    def anchors_campaign(self):
        """ Select the Most confidence bbox from the given anchors candidates in each grid unit. (1 / N_anc) """
        if not self.credible_grids:
            raise Exception("credible_grids is empty, plz execute the confidence_filter method first.")

        # Prepare selection mask through founding the best confidence
        conf_oh_masks = []                                  # onehot_mask {list(bool): 8} [(n_cred_m, n_anc), ...]
        for conf in self.credible_grids.get("conf"):        # conf {Tensor: (n_cred_m, n_anc(4))}
            conf_argmax  = torch.argmax(conf, dim=-1)       # (n_cred_m,)
            conf_mask = torch.stack([conf_argmax == ch for ch in range(self.preds.n_anchor)], -1)
            conf_oh_masks.append(conf_mask)

        # Selecting the bbox that have the best confidence in a grid.
        for attrib in self.credible_grids.keys():
            # attrib_val: {list: 8} [(n_cred_m, n_anc, n_attrib), ...]
            new_attrib_val = [attr_val[mask] for attr_val, mask in zip(self.credible_grids.get(attrib), conf_oh_masks)]
            self.credible_grids.update({attrib: new_attrib_val})  # update attributes

    def grid_non_maximum_suppression(self, **kwargs):
        """ Non-maximum suppression based-on IOU overlap threshold.
        Notice:This method is only works on the threshold filtered boxes, that means
        the prerequisite for this method is performing the 'confidence_filter' """
        # for each sample, generate the picked indexes through compute NMS
        pick_indexes = []
        gdsz = self.preds.grid_size
        properties = ("xy", "wh", "conf", "scores")
        for xy, wh, conf, scores in zip(*[self.credible_grids.get(item) for item in properties]):
            pick = non_maximum_suppress(conf, torch.cat([xy, wh], dim=-1), (gdsz, gdsz), **kwargs)
            pick_indexes.append(torch.tensor(pick, device=self.preds.device))

        # Update the credible grids with pick indexes generated from NMS
        for attr in self.credible_grids.keys():
            # attrib_val: {list: 8} [(n_cred_m, n_anc, n_attrib), ...]
            new_attrib_val = [attr_val[pick] for attr_val, pick in zip(self.credible_grids.get(attr), pick_indexes)]
            self.credible_grids.update({attr: new_attrib_val})

        # Adding an element 'class' representing the predicted object class to the credible_grids
        # According to the 'scores' for each object class, the object could be determined by the best one.
        obj_classes = [torch.argmax(score, dim=-1) for score in self.credible_grids["scores"]]
        self.credible_grids.update({"labels": obj_classes})

    def match_real_shape(self, image_shape):
        """ Scale the output data to match the original image size. """
        scales = torch.tensor(image_shape, device=self.preds.device)/self.preds.grid_size
        for attrib in ("xy", "wh"):
            # new_attrib_val: {list: 8} [(n_cred_m, n_anc, n_attrib), ...]
            new_attrib_val = [attrib_val*scales for attrib_val in self.credible_grids.get(attrib)]
            self.credible_grids.update({attrib: new_attrib_val})  # update attributes

    def rbc_area_properties(self, modalities, distance_thres=22, otsu_bins=20,
                            properties=("area", "bbox", "eccentricity", "perimeter", "intensity_image")):
        """ Extract recognized RBC cells with the dominant areas and
        produce specified properties into and pandas dict for every phase image.
        Returns
        -------
        properties_tables: list
            [properties_table_0: {'area', 'bbox', ...}, properties_table_1, ...]
        """
        def form_factor(image):
            return 4*np.pi*np.sum(image)/np.square(perimeter(image))

        def volume(image, intensity): return np.sum(intensity)

        def mean_phase_shift(image, intensity): return np.sum(intensity)/np.sum(image)

        def mch(image): return 10*WAVELENGTH*np.sum(image > 0)/(2*np.pi*0.002)

        # Extracting area labels
        rbc_labels = [single_sample_outlier_rbc(modalities[i][1], self[i].xy, self[i].wh, self[i].labels,
                                                distance_thres, otsu_bins) for i in range(len(self))]
        foreground_labels, background_labels = list(zip(*rbc_labels))

        # Properties Statistics, TODO: fill the hole in the cell center
        properties_tables = [regionprops_table(fore_label.astype(int), mod[1], properties,
                                               extra_properties=(form_factor, volume, mean_phase_shift, mch))
                             for fore_label, mod in zip(foreground_labels, modalities.numpy())]
        return properties_tables

    def statistic(self, out_dict):
        assert all([elm in out_dict for elm in ("rbc", "wbc", "platelet")])
        labels = torch.cat(self.credible_grids.get("labels"))
        (n_rbc, n_wbc, n_platelet), edges = np.histogram(labels, bins=3)
        out_dict["rbc"] += n_rbc
        out_dict["wbc"] += n_wbc
        out_dict["platelet"] += n_platelet

    def predict_accuracy(self):
        pass

    def predict_collection_show(self, images, focus="phase"):
        assert len(self.preds.data) == len(images)
        images = images.to("cpu").detach() if images.device.type == "cuda" else images.detach()
        # config artist assign
        properties = ("xy", "wh", "conf", "labels")
        n_row, n_col, figsize = [2, 4, (15, 7.5)] if len(images) == 8 else [2, 2, (8, 8)]
        fig, axis = figure_preproduction(figsize, n_row, n_col)
        for i, (mods, xy, wh, conf, labels) in enumerate(zip(images, *[self.get(item) for item in properties])):
            ax = axis[divmod(i, n_col)]
            ax.imshow(mods[1] if focus == "phase" else None, cmap="gray")
            for (x, y), (w, h), cf, lbl in zip(xy, wh, conf, labels):
                ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, fill=False, color=("green", "blue", "purple")[lbl]))
        plt.show()


class RealPhaseDataset(BloodSmearDataset):
    def __init__(self, **kwargs):
        super(RealPhaseDataset, self).__init__(**kwargs)
        self.image_transform = lambda img: torch.from_numpy(img).to(torch.float32)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        modalities_filenames = self.modalities_filenames[idx]
        # mod_filenames {tuple:4} (amp_name, pha_name, minus_name, plus_name)
        mod_fnames = [os.path.split(fp)[-1] for fp in modalities_filenames]
        assert len(set(mod_fnames)) == 1
        modalities = get_original_modality_from_id(mod_fnames[0])
        # Images and Labels Transform
        if self.image_transform:
            modalities = self.image_transform(modalities)
        if self.target_transform:
            labels = self.target_transform(labels)
        return modalities, labels


class Figures:
    @staticmethod
    def fig1a(region_properties_tables: list[dict]):
        """ region_properties_tables: List[{dict: 8}]
                attributes: 'area', 'bbox-0~3'(3), 'eccentricity', 'intensity_image', 'perimeter'
            for each attributes, contains many cells property values, for example:
                region_properties_tables[0].get('area')
                [area_00, area_01, ...] # area_00: the area for first RBC cell.
        """
        # Gathering all cells in different images into an single collection
        interest_properties = ("area", "eccentricity", "intensity_image", "perimeter",
                               "form_factor", "volume", "mean_phase_shift", "mch")
        gathered_properties_table = {
            key: np.concatenate([prop_table.get(key) for prop_table in region_properties_tables])
            for key in interest_properties
        }

        # Eccentricity-FormFactor Scatters
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.scatter(gathered_properties_table.get("area"), gathered_properties_table.get("volume"), s=0.5, c="blue")
        ax.set_xlabel("Area", fontsize=18)
        ax.set_ylabel("Volume", fontsize=18)
        # fig.savefig("..\\caches\\A-V_Demo.tif", dpi=200, bbox_inche="tight", pad_inches=0)
        plt.show()
        # Area-Volume Scatter
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.scatter(gathered_properties_table.get("eccentricity"),
                   gathered_properties_table.get("form_factor"), s=0.5, c="blue")
        ax.set_xlabel("Eccentricity", fontsize=18)
        ax.set_ylabel("Form Factor", fontsize=18)
        # fig.savefig("..\\caches\\Ec-ff_Demo.tif", dpi=200, bbox_inche="tight", pad_inches=0)
        plt.show()

    @staticmethod
    def fig1b(cell_nums_pred: dict, labels):
        """ cell prediction statistics.
        Parameters
        ---------
        cell_nums_pred: dict
            {"rbc": **, "wbc": **, "platelet": **}
        labels:
             [total_samples, N_labels,  x-y-w-h-l]]
        """
        labels_boxes = labels.view((-1, 5))  # [Total_labels,  x-y-w-h-l]
        truth_labels = torch.stack(list(filter(lambda box: torch.mean(box) > 0, labels_boxes)))[:, -1]  # [Total_labels]
        (n_rbc_truth, n_wbc_truth, n_platelet_truth), _ = np.histogram(truth_labels, bins=3)
        n_rbc_pred, n_wbc_pred, n_platelet_pred = [cell_nums_pred.get(item) for item in ("rbc", "wbc", "platelet")]
        rbc_acc = 100*(n_rbc_pred/n_rbc_truth)
        wbc_acc = 100*(n_wbc_pred/n_wbc_truth)
        plt_acc = 100*(n_platelet_pred/n_platelet_truth)
        rbc_err = 100*(abs(n_rbc_pred-n_rbc_truth)/n_rbc_truth)
        wbc_err = 100*(abs(n_wbc_pred-n_wbc_truth)/n_wbc_truth)
        plt_err = 100*(abs(n_platelet_pred-n_platelet_truth)/n_platelet_truth)

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.set_yscale("log")
        x = np.arange(3)  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax.bar(x - width / 2, (n_rbc_truth, n_wbc_truth, n_platelet_truth),
                        width, label='Truth', color="blue")
        rects2 = ax.bar(x + width / 2, (n_rbc_pred, n_wbc_pred, n_platelet_pred),
                        width, label='Prediction', color="red")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel('Cell Nums', fontsize=18)
        ax.set_ylim(1, 40000)
        ax.set_xticks(x)
        ax.set_xticklabels(("rbc", "wbc", "platelet"))

        ax.bar_label(rects1, padding=3, fontsize=12)
        ax.bar_label(rects2, padding=3, fontsize=12)

        ax2 = ax.twinx()
        # ax2.set_ylabel("Accuracy", rotation=270, fontsize=18)
        ax2.plot(x, (rbc_acc, wbc_acc, plt_acc))
        ax2.set_ylim(0, 100)
        plt.show()


def image_recognition(dataloader, model_fpath, pred_params, reproduce=False):
    """ Testing the trained network models."""
    cache_filename = ".\\caches\\TestSet_RBC_RegionProperties.pkl"

    def _batch_phase_normalize(inputs):
        outputs = deepcopy(inputs)
        _phase = outputs[:, 1]
        outputs[:, 1] = torch.sigmoid((_phase-torch.mean(_phase))/torch.std(_phase))
        return outputs

    # Read cached files
    cells_num_preds = {"rbc": 0, "wbc": 0, "platelet": 0}
    region_properties, predict_handles, y_data = [], [], []
    if os.path.isfile(cache_filename) and not reproduce:
        logger.info("Loading Cached files")
        with open(cache_filename, "rb") as f:
            region_properties, predict_handles, y_data, cells_num_preds = pickle.load(f)
    else:
        logger.info("Loading Model")
        yolo_model = torch.load(model_fpath).to("cuda")
        # Data Proceeding
        for modalities, labels in tqdm.tqdm(dataloader):
            # turns all data to image type (~[0, 1])
            model_input = _batch_phase_normalize(modalities)
            # Model FP prediction
            model_output = yolo_model(model_input.to("cuda")).to("cpu").detach_()
            # Processing model output
            pred_handle = PredHandle(model_output, **pred_params)
            # Cell statistics
            pred_handle.statistic(cells_num_preds)
            # pred_handle.predict_collection_show(modalities)
            # region_properties += pred_handle.rbc_area_properties(modalities)
            # Gather results
            predict_handles.append(pred_handle)
            y_data.append(labels)

        # Save region properties dict into disk
        with open(cache_filename, "wb") as f:
            pickle.dump((region_properties, predict_handles, y_data, cells_num_preds), f)

    # Plot figures
    # Figures.fig1a(region_properties)
    Figures.fig1b(cells_num_preds, torch.cat([lbl[3] for lbl in y_data], dim=0))


if __name__ == '__main__':
    # ================== Python interpreter initializing =================
    if not plt.get_backend() == "tkagg":
        plt.switch_backend("tkagg")
    # Script Constants
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SINGLE_SAMPLE_SET = join(DATA_ROOT, "SingleBatchSet")

    # Logging Config ---------------------------------------------------
    logging.config.fileConfig(".\\log\\config\\evaluate.conf")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # ========================== Scripts Execute =========================
    try:
        confThres   =   0.6
        overThres   =   0.3
        pipeDevice  =   "cpu"
        modelName   =   join(MODEL_PATH, "plan_8.0\\yolov5_1020-153749.pth")
        testLoader  =   create_dataloader(SET202109_FILENAME, "test", 8, dataset_obj=RealPhaseDataset, shuffle=True)

        predParams = {"image_shape": (DST_IMGSZ, DST_IMGSZ), "grid_size": GRIDSZ,
                      "anchors": ANCHORS, "n_class": N_CLASSES, "device": pipeDevice,
                      "nms_overlapping_thres": overThres, "conf_thres": confThres}
        image_recognition(testLoader, modelName, pred_params=predParams)

    except Exception as e:
        logger.exception(e)
