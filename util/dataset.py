import os
import re
import types
import json
import yaml
import pickle
import shutil
import logging.config

from os import listdir
from functools import partial
from pprint import pformat
from collections import UserDict
from xml.dom.minidom import parseString
from typing import List, Optional
from xml.etree import ElementTree as ET
from os.path import join, split, isfile, basename, isdir, splitext, exists, dirname

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cv2 import resize
from scipy.io import savemat
from numpy import quantile
from matplotlib import figure
from skimage.filters import gaussian
from skimage.io import imread, imsave
from torch.utils.data import Dataset, DataLoader
from skimage.util import dtype_limits, img_as_ubyte, img_as_float32, crop, invert

from ..util.functions import image_baseline, load_txt_list
from cccode.tie_tech import tie_solution, energy_match, tie_solve_differential, MatlabRegister
from cccode.image import LowFrequencyFilter, phasermic_imsplit, defocus_background_estimate, ck
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DST_IMGSZ = 320
SRC_IMGSZ = 340
GRIDSZ = 40
N_CLASSES = 3
IMG_PLUGIN = "simpleitk"
F32 = torch.float32
ANCHORS = [1., 1., 1.125, 1.125, 1.25, 1.25, 1.375, 1.375]
DATA_ROOT = "D:\\Workspace\\Blood Recognition\\datasets"
SOURCE_DATASETS = ['20210902 BloodSmear01', '20210902 BloodSmear02',
                   '20210914 BloodSmear01', '20210914 BloodSmear02']

# Pytorch format Dataset Constructor, using in the initialing of 'BloodSmearDataset'
SET202109_FILENAME = join(DATA_ROOT, "Set-202109-1")
SINGLE_SAMPLE_SET = join(DATA_ROOT, "SingleBatchSet")
XML_DATASET_FILENAME = join(DATA_ROOT, "20210105 BloodSmear\\fov_annotations.xml")


def image_split(image, chunk_size, n_split=3):
    h_image, w_image = image.shape
    h_chunk, w_chunk = chunk_size
    centroid_x_arrange = w_image - w_chunk
    centroid_y_arrange = h_image - h_chunk
    centroid_x_interval = centroid_x_arrange // (n_split - 1)
    centroid_y_interval = centroid_y_arrange // (n_split - 1)

    subviews = []
    for i in range(n_split):  # ROW split
        for k in range(n_split):  # COLUMN split
            x_centroid = w_chunk // 2 + k * centroid_x_interval
            y_centroid = h_chunk // 2 + i * centroid_y_interval
            x0, y0 = x_centroid - w_chunk // 2, y_centroid - h_chunk // 2
            x1, y1 = x_centroid + w_chunk // 2, y_centroid + h_chunk // 2
            subviews.append(image[y0:y1, x0:x1])
    return subviews


class MultimodalSample:
    """ labels: [(cls, x, y, w, h)] """

    def __init__(self):
        self.sample_idx = None
        self.phase = None
        self.amplitude = None
        self.overfocus = None
        self.underfocus = None
        self.labels = []
        self.source_msg = {}
        self.image_shape = None

    def __repr__(self):
        output = f"<Ojb MultimodalSample>\n"
        if self.sample_idx is not None:
            output += f"\timage_index:    {self.sample_idx:<4d}\n"

        names = ["amplitude", "phase", "under-focus", "over-focus"]
        str_max_len = np.max([len(n) for n in names]) + 3
        for modality, name in zip(self.modalities, names):
            output += "\t" + "{name}:".ljust(str_max_len) + f"{type(modality)}, {modality.shape.__repr__()}," \
                                                            f"val_range: {dtype_limits(modality)}\n"
        output += f"\tlabel numbers:  {len(self.labels)}\n\tsource_messages:\n"
        output += pformat(self.source_msg, indent=8)
        return output

    @property
    def modalities(self):
        return self.amplitude, self.phase, self.underfocus, self.overfocus

    def set_modalities(self, amp, pha, under, over):
        self.amplitude, self.phase, self.underfocus, self.overfocus = amp, pha, under, over

    @classmethod
    def from_element(cls, sample: ET.Element, read_image=True):
        mm_sample = cls()
        amp_fullname = sample.find("amp_fullname").text
        pha_fullname = sample.find("pha_fullname").text
        over_fullname = sample.find("over_fullname").text
        under_fullname = sample.find("under_fullname").text

        mm_sample.source_msg.update({"amp_fullname": amp_fullname, "pha_fullname": pha_fullname,
                                     "over_fullname": over_fullname, "under_fullname": under_fullname})
        mm_sample.sample_idx = int(sample.find("image_idx").text)

        if read_image:
            mm_sample.amplitude = imread(amp_fullname, True, "simpleitk")
            mm_sample.phase = imread(pha_fullname, True, "simpleitk")
            mm_sample.overfocus = imread(over_fullname, True, "simpleitk")
            mm_sample.underfocus = imread(under_fullname, True, "simpleitk")

        img_shape_elm = sample.find("image_shape")
        if img_shape_elm is not None:
            height, width = [int(img_shape_elm.find(tag).text) for tag in ["height", "width"]]
            mm_sample.image_shape = (height, width)
            if mm_sample.amplitude is not None:
                assert (height, width) == mm_sample.amplitude.shape

        labels_elm = sample.find("labels")
        for lbl_elm in labels_elm.findall("label"):
            bbox_elm = lbl_elm.find("bbox")
            x, y, w, h = [int(bbox_elm.find(tag).text) for tag in ("x", "y", "w", "h")]
            lbl_cls = int(lbl_elm.find("class").text)
            mm_sample.labels.append((lbl_cls, x, y, w, h))
        return mm_sample

    @staticmethod
    def split(sample, n_split=3, target_size=(320, 320)):
        assert (sample.phase.shape == sample.amplitude.shape == sample.underfocus.shape == sample.overfocus.shape)
        height, width = sample.phase.shape
        tgt_height, tgt_width = target_size
        centroid_arrange_width = width - tgt_width
        centroid_arrange_height = height - tgt_height

        centroid_interval_width = centroid_arrange_width // (n_split - 1)
        centroid_interval_height = centroid_arrange_height // (n_split - 1)

        subview_multimodal_samples: List[MultimodalSample] = []
        for i in range(n_split):  # ROW split
            for k in range(n_split):  # COLUMN split
                x_centroid = tgt_width // 2 + k * centroid_interval_width
                y_centroid = tgt_height // 2 + i * centroid_interval_height

                x0 = x_centroid - tgt_width // 2
                x1 = x_centroid + tgt_width // 2
                y0 = y_centroid - tgt_height // 2
                y1 = y_centroid + tgt_height // 2

                subview_modalities = [image[y0:y1, x0:x1] for image in sample.modalities]

                subview_labels = []
                for cls, x, y, w, h in sample.labels:
                    if x0 <= x < x1 and y0 <= y < y1:
                        subview_labels.append((cls, x - x0, y - y0, w, h))

                subview_mlt_sample = MultimodalSample()
                subview_mlt_sample.labels = subview_labels
                subview_mlt_sample.source_msg.update({"parent_msg": sample.source_msg})
                subview_mlt_sample.set_modalities(*subview_modalities)
                subview_multimodal_samples.append(subview_mlt_sample)
        return subview_multimodal_samples

    def save(self, save_root):
        amp_path, pha_path, minus_path, plus_path, anno_path = [join(save_root, pth) for pth in
                                                                ("amp", "pha", "minus", "plus", "anno")]
        for pth in (amp_path, pha_path, minus_path, plus_path, anno_path):
            if not os.path.exists(pth):
                os.mkdir(pth)

        amp_fullname = join(amp_path, f"amp_{self.sample_idx:04d}.jpg")
        pha_fullname = join(pha_path, f"pha_{self.sample_idx:04d}.jpg")
        minus_fullname = join(minus_path, f"minus_{self.sample_idx:04d}.jpg")
        plus_fullname = join(plus_path, f"plus_{self.sample_idx:04d}.jpg")
        anno_fullname = join(anno_path, f"pha_{self.sample_idx:04d}.txt")

        # save images
        for modality, fname in zip(self.modalities, (amp_fullname, pha_fullname, minus_fullname, plus_fullname)):
            # amplitude, phase, underfocus, overfocus
            imsave(fname, img_as_ubyte(modality), "simpleitk")

        # save labels
        yolo_string = ""
        height, width = self.amplitude.shape
        for cell_class, x, y, w, h in self.labels:
            # Coordinate transform from absolute coordination to yolo
            x1, y1 = x / width, y / height
            w1, h1 = w / width, h / height
            coord_str = " ".join([f"{coord:1.6f}" for coord in (x1, y1, w1, h1)])
            yolo_string += str(cell_class) + " " + coord_str + "\n"

        # save single sample yolo file
        with open(anno_fullname, "w") as f:
            f.write(yolo_string)

    def out_labeled_data(self, dst_image_shape: tuple, modalities_to=None, labels_to: Optional[list] = None):
        # cast image and labels to destined shape
        if self.image_shape is None:
            self.image_shape = self.modalities[0].shape

        if self.image_shape != dst_image_shape:
            self.set_modalities(*[resize(img, dst_image_shape) for img in self.modalities])
            for i, lbl in enumerate(self.labels):
                scale_x = dst_image_shape[1] / self.image_shape[1]
                scale_y = dst_image_shape[0] / self.image_shape[0]
                cls, x, y, w, h = lbl
                x, w = x * scale_x, w * scale_x
                y, h = y * scale_x, h * scale_y
                self.labels[i] = cls, x, y, w, h
        if modalities_to and labels_to:
            modalities_to.append(np.array(self.modalities))
            labels_to.append(np.array(self.labels))
        else:
            return np.array(self.modalities), np.array(self.labels)

    def annotate_axes(self, ax: mpl.figure.Axes):
        for _, x, y, w, h in self.labels:
            rect = plt.Rectangle((x - w // 2 - 1, y - h // 2 - 1), w, h, fill=False, color="blue")
            ax.add_patch(rect)
        return ax


class StandardXMLContainer:
    """ For each child sample, there are six SubElements:
            'image_shape', "image_idx', 'amp_fullname', 'pha_fullname',
            'under_fullname', 'over_fullname', and 'labels'
    The 'labels' stored all automatic annotated labels inside this sample,
    as the child element of 'labels'. """

    def __init__(self):
        self.root = ET.Element("FFoV_Annotation")
        self.sample_elements = None

    @classmethod
    def from_xml(cls, filename):
        container: StandardXMLContainer = cls()
        etree = ET.parse(filename)
        container.root = etree.getroot()
        container.sample_elements = container.root.findall("sample")
        container.src_filename = filename
        return container

    def sample_collection_visualization(self, start_idx=0, figsize=(20, 12), title=None, tt_fontsize=16):
        visualized_samples = [MultimodalSample.from_element(sp) for sp in
                              self.sample_elements[start_idx:start_idx + 15]]
        fig, axs = plt.subplots(3, 5, figsize=figsize, constrained_layout=True)
        if title is not None:
            plt.suptitle(title, fontproperties={"size": tt_fontsize})
        for row in range(3):
            for collum in range(5):
                n_sp = row * 5 + collum
                ax = axs[row, collum]
                sample: MultimodalSample = visualized_samples[n_sp]
                ax.imshow(sample.phase, cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ("top", "bottom", "left", "right"):
                    ax.spines[sp].set_visible(False)
                sample.annotate_axes(ax)
        plt.show()

    @staticmethod
    def sub_elem(parent: ET.Element, child_tag: str, text: str = None):
        child = ET.SubElement(parent, child_tag)
        if text is not None:
            child.text = text if isinstance(text, str) else str(text)
        return child

    @staticmethod
    def sample_msg(sample: ET.Element):
        img_idx_str = sample.find('image_idx').text
        height, width = [int(sample.find("image_shape").find(tag).text) for tag in ("height", "width")]
        output = f"Sample\n\timage idx:      {img_idx_str}\n\timage shape:    ({height}, {height})\n"

        amp_fpath, amp_fname = split(sample.find("amp_fullname").text)
        pha_fpath, pha_fname = split(sample.find("pha_fullname").text)
        over_fpath, over_fname = split(sample.find("over_fullname").text)
        under_fpath, under_fname = split(sample.find("under_fullname").text)
        amp_root, amp_dir = split(amp_fpath)
        pha_root, pha_dir = split(pha_fpath)
        over_root, over_dir = split(over_fpath)
        under_root, under_dir = split(under_fpath)
        try:
            assert amp_root == pha_root == over_root == under_root
        except AssertionError as e:
            logger.exception(e)

        num_labels = len(sample.find("labels").findall("label"))
        output += f"\tmodalities root: {amp_root}\n"
        output += f"\t\tamplitude     dir: {amp_dir:<6s} filename: {amp_fname}\n"
        output += f"\t\tphase         dir: {pha_dir:<6s} filename: {pha_fname}\n"
        output += f"\t\tunder-focus   dir: {over_dir:<6s} filename: {under_fname}\n"
        output += f"\t\tover-focus    dir: {under_dir:<6s} filename: {over_fname}\n"
        output += f"\tlabels number:  {num_labels}\n"
        return output

    def add_sample(self, idx, image_shape, amplitude_filename, phase_filename, minus_filename, plus_filename):
        sample = ET.SubElement(self.root, "sample")
        shape = self.sub_elem(sample, "image_shape")
        self.sub_elem(shape, "height", image_shape[0])
        self.sub_elem(shape, "width", image_shape[1])

        self.sub_elem(sample, "image_idx", str(idx))
        self.sub_elem(sample, "amp_fullname", amplitude_filename)
        self.sub_elem(sample, "pha_fullname", phase_filename)
        self.sub_elem(sample, "over_fullname", plus_filename)
        self.sub_elem(sample, "under_fullname", minus_filename)
        self.sub_elem(sample, "labels")
        return sample

    def add_label(self, sample: ET.Element, x, y, w, h, cell_class, creator: str):
        node = sample.find("labels")
        label = ET.SubElement(node, "label")
        bbox = ET.SubElement(label, "bbox")
        self.sub_elem(label, "class", str(cell_class))
        self.sub_elem(label, "creator", creator)

        # add all bounding box values
        assert isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int)
        self.sub_elem(bbox, "x", x)
        self.sub_elem(bbox, "y", y)
        self.sub_elem(bbox, "w", w)
        self.sub_elem(bbox, "h", h)

    def compile(self, fpath: str = None, prettify=True):
        xml_string = ET.tostring(self.root, encoding="utf-8")
        if prettify:
            reparsed = parseString(xml_string)
            xml_string = reparsed.toprettyxml(indent="\t")

        with open(fpath, "w") as f:
            f.write(xml_string)

    def toyolo(self, yolopath, prefix="phase"):
        yolo_pattern = "phase_{:02d}.txt" if prefix == "phase" else "img_{:02d}.txt"

        for sample in self.root.findall("sample"):
            img_idx = int(sample.find("image_idx").text)
            image_shape = sample.find("image_shape")
            height, width = [int(image_shape.find(tag).text) for tag in ("height", "width")]

            yolo_string = ""
            for lbl in sample.find("labels").findall("label"):
                bbox = lbl.find("bbox")
                x, y, w, h = [int(bbox.find(tag).text) for tag in ("x", "y", "w", "h")]
                cell_class = lbl.find("class").text

                # Coordinate transform from absolute coordination to yolo
                x1, y1 = x / width, y / height
                w1, h1 = w / width, h / height
                coord_str = " ".join([f"{coord:1.6f}" for coord in (x1, y1, w1, h1)])
                yolo_string += cell_class + " " + coord_str + "\n"

            # save single sample yolo file
            dst_fullname = join(yolopath, yolo_pattern.format(img_idx))
            with open(dst_fullname, "w") as f:
                f.write(yolo_string)

    def sample_slicing(self, split_root="", sample_per_batch=10, dst_shape=(340, 340)) -> List[MultimodalSample]:
        """ Slicing all samples into subview samples """
        batch_path = []
        sample_set = []
        n_samples = len(self.sample_elements)

        if split_root != "":  # Preparing for storage
            batch_nums = n_samples // sample_per_batch + 1
            batch_path = [join(split_root, f"batch_{n_bth:02d}") for n_bth in range(batch_nums)]
            for pth in [split_root] + batch_path:
                if not os.path.exists(pth):
                    os.mkdir(pth)

        for i, sp in enumerate(self.sample_elements):
            sp = MultimodalSample.from_element(sp)  # XML <Elements> to <MultimodalSample>
            children = MultimodalSample.split(sp, target_size=dst_shape)
            for k, child in enumerate(children):
                child.sample_idx = i * 9 + k
                if split_root:  # Perform storage
                    save_root = batch_path[i // sample_per_batch] if sample_per_batch else split_root
                    child.save(save_root)
            sample_set += children
        return sample_set

    def dataset_output(self, destined_image_shape, splitting=True):
        # Create a container for parsing the raw data loaded from disk,
        # and operating (Splitting) the dataset through attached method.
        # sample_sets: np.array([MltSample1, MltSample2, MltSample3, ....])
        sample_sets = self.sample_slicing() if splitting else None

        modalities_set, labels_set = [], []
        for i, subviewSample in enumerate(sample_sets):
            subviewSample.out_labeled_data(destined_image_shape, modalities_to=modalities_set, labels_to=labels_set)

        # Determine the maximum labels number
        max_boxes = max([len(lbl) for lbl in labels_set])

        # Packaging the labels using uniform size array: [N_samples, n_label_per_sample, 5]
        boxes_set = np.zeros((len(labels_set), max_boxes, 5))
        for i, label in enumerate(labels_set):  # overwrite the N boxes_set info  [N,5]
            boxes_set[i, :label.shape[0]] = label
        return modalities_set, boxes_set


def dataset_xml_from_annotations(focus_path, phase_path, minus_path, plus_path, annotations_path,
                                 xml_filename, imgfmt=".bmp", fixed_class="rbc", creator="auto"):
    """ Now that the modality and tag data have been generated, it is necessary to combine
    these data into a single xml file to organize the subsequent training dataset.
    """
    minus_filenames, plus_filenames, focus_filenames, phase_filenames, ann_filenames = [
        [fname for fname in os.listdir(pth) if fname.endswith(imgfmt, ".txt")]
        for pth in (minus_path, plus_path, focus_path, phase_path, annotations_path)]

    # create root xml Element
    xml_container = StandardXMLContainer()
    for *fnames, ann_fname in zip(focus_filenames, phase_filenames, minus_filenames, plus_filenames, ann_filenames):
        indexes = [fname.rstrip(imgfmt).lstrip(pref) for fname, pref in
                   zip(fnames, ("focus_", "phase_", "minus_", "plus_"))]
        ann_idx = ann_fname.removesuffix(".txt").removeprefix("auto_")

        try:
            assert all([ann_idx == idx for idx in indexes])
        except AssertionError as e:
            logger.exception(e)

        fullnames = [join(ph, fn) for ph, fn in zip((focus_path, phase_path, minus_path, plus_path), fnames)]
        ann_fullname = join(annotations_path, ann_fname)

        # add sample
        img_idx = int(ann_idx)
        image_shape = imread(fullnames[0], True, "simpleitk").shape  # load image shape
        sample = xml_container.add_sample(img_idx, image_shape, *fullnames)
        # Loading the annotations
        with open(ann_fullname, "r") as f:
            for line in f:
                if line.endswith("\n"):
                    line = line.rstrip("\n")
                num_strings = list(filter(lambda elm: elm != "", line.split(" ")))
                x, y, w, h = [int(num) for num in num_strings]

                if fixed_class == "rbc":  # Add label
                    clsn = 0
                xml_container.add_label(sample, x, y, w, h, clsn, creator)

    # writes into disk
    xml_container.compile(xml_filename)
    return xml_container


def get_original_modality_from_id(input_id, dataset_root=DATA_ROOT, shift_baseline=True,
                                  subview_shape=(340, 340), dst_shape=(320, 320)):
    # If id with suffix, remove it
    for suf in (".txt", ".jpg"):
        input_id = input_id.removesuffix(suf)

    # Analysis the image belong to, e.g. 20210902-20411.jpg
    # -> dataset_folder: 20210902 BloodSmear02, image_id: 45 (411//9), patch_id: 6 (411%9)
    acquired_date, picture_id = input_id.split("-")
    folder_id, subview_id = int(picture_id[:1]), int(picture_id[1:])
    image_id, patch_id = divmod(subview_id, 9)

    dataset_folder = f"{acquired_date:s} BloodSmear{folder_id:02d}"
    dataset_fpath = join(dataset_root, dataset_folder)

    cache_path = join(dataset_fpath, "ndarrays")
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    cache_name = join(cache_path, f"{image_id:02d}.pkl")

    try:  # Try read from caches
        with open(cache_name, "rb") as f:
            modality = pickle.load(f)

    except IOError:  # No Caches, Create it
        remove_strips = False
        try:  # Read raw defocus(acquired by the camera, without processing), No background removing
            defocus_folder = join(dataset_fpath, "raw_images")
            bs_handle = BloodSmearImageHandel(filename=join(defocus_folder, f"{image_id:02d}.bmp"))
            # Eliminate inhomogeneous illumination
            bs_handle.balance_illumination(backgrounds_folder=join(dataset_fpath, "backgrounds"))
            # Image FoV registering
            bs_handle.align_fov_mismatch(error_fname=join(defocus_folder, f"mismatch_{image_id // 10:02d}.pkl"))
            # Match Energy
            bs_handle.energy_matching()

            if remove_strips:  # Remove Strips
                lf_filter = partial(LowFrequencyFilter.interest_filtering, interest_radius=8)
                minus_corr, plus_corr = list(map(lf_filter, (bs_handle.minus, bs_handle.plus)))
                correction = (bs_handle.minus - minus_corr + invert(bs_handle.plus - plus_corr)) / 2
                correction_norm = correction - np.mean(correction)
                bs_handle.minus, bs_handle.plus = bs_handle.minus - correction_norm, bs_handle.plus + correction_norm

            minus, plus = bs_handle.minus, bs_handle.plus
            focus = (minus + plus) / 2

        except RuntimeError:  # Couldn't get raw image, load from the processed defocus images.
            minus_filename = join(dataset_fpath, f"minus\\minus_{image_id:04d}.bmp")
            plus_filename = join(dataset_fpath, f"plus\\plus_{image_id:04d}.bmp")
            minus, plus = imread(minus_filename, True, "simpleitk"), imread(plus_filename, True, "simpleitk")
            focus = (minus + plus) / 2

        phase = tie_solution(focus, minus, plus, 1e-3, 532e-9, 4.8e-6, 3e7)
        # Flatten phase background
        if shift_baseline:
            phase -= image_baseline(phase)

        # Split and return subview multi-modality
        modality = (focus, phase, minus, plus)
        with open(cache_name, "wb") as f:
            pickle.dump(modality, f)

    modality_subviews = [image_split(mod, subview_shape, 3) for mod in modality]
    required_subviews = [subview[patch_id] for subview in modality_subviews]
    if dst_shape:
        required_subviews = [resize(mod, dst_shape) for mod in required_subviews]
    return np.stack(required_subviews)


class DatasetStructure:
    """Manage and Utilizing the Dataset"""

    def __init__(self, root, excludes=("CVAT SourceData-SplitSamples",)):
        self.root = self.scan_tree(root, lambda x: x.startswith(("202109", "Set-")), excludes)

    @staticmethod
    def _get_child(parent: dict, child_name):
        for child in parent.get("children"):
            if isinstance(child, UserDict):  # a directory
                if child.get("name") == child_name:
                    return child
            elif isinstance(child, str):  # a file
                if child == child_name:
                    return child
        return None

    def scan_tree(self, root, interest_filter, exclude_names):
        interest_names = list(filter(interest_filter, listdir(root)))
        def path_filter(x): return x not in exclude_names
        data = UserDict({"name": basename(root), "fullpath": root,
                         "children": [self.scan_directory(join(root, fn), path_filter)
                                      if isdir(join(root, fn)) else join(root, fn) for fn in interest_names]})
        data.get_child = types.MethodType(self._get_child, data)
        return data

    def scan_directory(self, fullpath: str, path_filter=tuple):
        interest_names = list(filter(path_filter, listdir(fullpath)))
        data = UserDict({"name": basename(fullpath), "fullpath": fullpath,
                         "children": [self.scan_directory(join(fullpath, fn), path_filter)
                                      if isdir(join(fullpath, fn)) else join(fullpath, fn) for fn in interest_names]})
        data.getChild = types.MethodType(self._get_child, data)
        return data

    def __repr__(self):
        def _presents(elm, indent=0):
            str_body = "\t" * indent + "\\" + elm["name"] + "\n"
            for child in elm["children"]:
                if isinstance(child, UserDict) and len(child):
                    str_body += _presents(child, indent + 1)
            return str_body
        return "." + _presents(self.root)

    def subset_phase_replacing(self, subset_name):
        """ Replacing the terrible phase results of the previous solution.
        Date: 2021-11-08
        """
        subset = self.root.get_child(subset_name)
        phase_matrix_path = join(subset.get("fullpath"), "phase_matrix")
        if not exists(phase_matrix_path):
            os.mkdir(phase_matrix_path)
        phase_path = subset.get_child("phase")["fullpath"]
        background_fpath = subset.get_child("backgrounds")["fullpath"]
        raw_image_filenames = subset.get_child("raw_images")["children"]
        error_fname_pattern = join(subset.get_child("raw_images")["fullpath"], "mismatch_%02d.pkl")
        sparse_indexes = load_txt_list(join(subset.get_child("raw_images")["fullpath"], "sparse_indexes.txt"))
        for i in range(4, 20):
            fullname = raw_image_filenames[i]
            img_name = basename(fullname)
            img_id = int(splitext(img_name)[0])
            error_fname = error_fname_pattern % (img_id // 10)
            dst_phase_filename = join(phase_path, f"phase_{img_id:04d}.bmp")
            phase_new = BloodSmearImageHandel.processing(fullname, background_fpath, error_fname,
                                                         otsu_radii=50 if img_id not in sparse_indexes else 120)
            # Comparing
            old_phase = imread(dst_phase_filename)
            ck.img_show(old_phase, phase_new, names=('old', 'new'), title=f"ID {img_id:02d}")
            # Save
            dst_array_filename = join(phase_matrix_path, f"phase_{img_id:04d}.mat")
            savemat(dst_array_filename, {"phase": img_as_float32(phase_new)})
            imsave(dst_phase_filename, phase_new)

    @staticmethod
    def dataset_reproduce(root, source_subset_names, labels_fpath, dst_dataset,
                          modality_flags=("focus", "phase", "minus", "plus"),
                          chunk_size=(340, 340), n_split=3, probabilities=(.8, .1, .1)):
        """ Update the dataset for better model training using the reproduced phase results.
        From:
            amplitude:      $SUBSET$/focus/focus_****.bmp
            phase:          $SUBSET$/phase/phase_****.bmp
            under-focus:    $SUBSET$/minus/minus_****.bmp
            over-focus:     $SUBSET$/plus/plus****.bmp
            +++++++++++++++++++++++++++++++++++++++++++
            labels:         'CVAT Caches/labels collection/$BATCH_NAME$/anno/pha_****.txt'
        To:
            new_set:        $NEW_SET$/$SUBSET_TYPE$/$MOD_NAME$/$FILE_NAME$
        Date: 2021-11-08
        """
        # load labels
        def _parse_label_batch_name(batch_name, subset_pattern="%s BloodSmear%02d"):
            """ Sample demo: 20210902-1_batch0 """
            subset_date, subset_id, _batch_id = re.split(r"[-_a-z]+", batch_name)
            return subset_pattern % (subset_date, int(subset_id)), _batch_id

        def _parse_label_name(label_name, _pref, _suff):
            lbl_id = label_name.removesuffix(_suff).removeprefix(_pref)
            _src_img_id, _src_img_patch = divmod(int(lbl_id), 9)
            return _src_img_id, _src_img_patch

        # Analysis the all labels files, determines the corresponding source images, output the message list
        labels_messages = []
        lbl_pref, lbl_suff = "pha_", ".txt"
        for lbl_batch in listdir(labels_fpath):
            corr_subset, _ = _parse_label_batch_name(lbl_batch)
            for f in listdir(join(labels_fpath, lbl_batch, "anno")):
                if f.startswith(lbl_pref) and f.endswith(lbl_suff):
                    lbl_fullname = join(labels_fpath, lbl_batch, "anno", f)
                    corr_img_id, corr_img_patch = _parse_label_name(f, lbl_pref, lbl_suff)
                    labels_messages.append({"subset": corr_subset,
                                            "img_id": corr_img_id,
                                            "img_patch": corr_img_patch,
                                            "lbl_fullname": lbl_fullname})

        source_images_container = {subset_name: {} for subset_name in source_subset_names}

        def _modality_chunks_from_imid(images_container, _subset, _imid, _impt, _mod_fullnames):
            _mod_names, _ = list(zip(*_mod_fullnames))
            subset_content = images_container[_subset]
            if img_dict := subset_content.get(str(_imid), None):
                # find cached images in container, call them and logs the used patch id, output the log info
                _modality_chunks = tuple([img_dict.get(_mod)[_impt] for _mod in _mod_names])
                img_dict["used_ids"] = img_dict["used_ids"] | {_impt}
                if len(img_dict["used_ids"]) == 9:
                    subset_content.pop(str(_imid))
                    logger.info(f"Image {_subset}-{_imid} deleted")
            else:
                # No cached images in the container, loads from disk and caches them
                named_chunks_tuple = [(flg, tuple(image_split(imread(fn, True, "simpleitk"), chunk_size, n_split)))
                                      for flg, fn in _mod_fullnames]
                updates = {str(_imid): dict({k: v for k, v in named_chunks_tuple}, **{"used_ids": {_impt}})}
                subset_content.update(updates)
                _modality_chunks = tuple([tp[1][_impt] for tp in named_chunks_tuple])
                logger.debug(updates[str(_imid)].keys())

            # Assign modality chunk names
            _setdate, _set_id = re.split(r" \w{10}", _subset)
            assigned_chunk_names = [f"{_setdate}-{int(_set_id)}{(_imid * 9) + _impt:04d}.bmp" for _mod in _mod_names]
            return _modality_chunks, assigned_chunk_names

        # load modalities
        sample_packages = []
        for msg_dict in labels_messages:
            subset, imid, impt, lbl_fullname = [msg_dict[k] for k in ("subset", "img_id", "img_patch", "lbl_fullname")]
            # get the corresponding modalities fullnames of this label
            mod_fullnames = [(flg, join(root, subset, flg, f"{flg}_{imid:04d}.bmp")) for flg in modality_flags]
            # loads the four modalities from the disk, and save them into the temporary container
            chunks, ck_names = _modality_chunks_from_imid(source_images_container, subset, imid, impt, mod_fullnames)
            # packaging sample
            setdate, set_id = re.split(r" \w{10}", subset)
            sample = dict({"label": (f"{setdate}-{int(set_id)}{(imid * 9) + impt:04d}.txt", lbl_fullname),
                           "message": {"subset": subset, "img_id": imid, "img_path": impt}},
                          **{mod: (fn, data) for mod, fn, data in zip(modality_flags, ck_names, chunks)})
            sample_packages.append(sample)

        # files destine
        train_stages = ("train", "valid", "test")
        dsst_modality_folders = "label", *modality_flags
        for mod in dsst_modality_folders:
            if not exists(dst_mod := join(dst_dataset, mod)):
                os.mkdir(dst_mod)
            for stage in train_stages:
                if not exists(dst_stage_fpath := join(dst_dataset, mod, stage)):
                    os.mkdir(dst_stage_fpath)

        assign_messages = []
        for sp in sample_packages:
            dst_stage = np.random.choice(train_stages, p=probabilities)
            for key, val in sp.items():
                if key != "message":  # modality data
                    dst_fullname = join(dst_dataset, key, dst_stage, val[0])
                    if key == "label":
                        shutil.copy(val[1], dst_fullname)
                    else:
                        imsave(dst_fullname, val[1])
                else:  # logging data
                    val.update({"stage": dst_stage})
                    assign_messages.append(val)

        with open(join(dst_dataset, "assignments.json"), "w") as f:
            json.dump(assign_messages, f, indent=4)


class BloodSmearImageHandel:
    """ Loading, pre-processing the blood smear defocus images acquired by bio-chips CMOS camera,
    and calculate the phase image for the next cell recognition algorithms. """

    def __init__(self, filename, fn_preprocess=None):
        self.filename = filename
        self.jointed_defocus = imread(filename, as_gray=True, plugin="simpleitk")
        if fn_preprocess:
            self.jointed_defocus = fn_preprocess(self.jointed_defocus)
        self.phase = None
        self.minus, self.plus = phasermic_imsplit(self.jointed_defocus)

    @property
    def difference(self): return self.plus - self.minus

    @property
    def focus(self): return (self.plus + self.minus) / 2

    def solve_phase(self, **tie_params):
        delta_d = tie_params.pop("delta_d")
        derivative = self.difference / (2 * delta_d)
        self.phase = tie_solve_differential(self.focus, derivative, **tie_params)

    def energy_matching(self):
        self.minus, self.plus = energy_match(self.minus, self.plus)

    def balance_illumination(self, backgrounds_folder):
        # Remove background caused by inhomogeneous illumination
        if isfile(backgrounds_filename := join(backgrounds_folder, "average.pkl")):
            with open(backgrounds_filename, "rb") as f:
                background_average = pickle.load(f)
        else:
            background_average = defocus_background_estimate(backgrounds_folder)

        # Remove background
        bg_minus, bg_plus = [gaussian(img, sigma=5) for img in phasermic_imsplit(background_average)]
        self.minus /= bg_minus / np.mean(bg_minus)
        self.plus /= bg_plus / np.mean(bg_plus)

    def phase_background_remove(self, otsu_radii=50):
        # phase = np.clip(self.phase, quantile(self.phase, 0.01), quantile(self.phase, 0.999))
        # self.phase = phase - background_estimate(phase, radius=0.04)
        phase_bg = image_baseline(self.phase, otsu_disk_radii=otsu_radii)
        self.phase -= phase_bg
        self.phase = np.clip(self.phase, quantile(self.phase, 0.0003), quantile(self.phase, 0.9997))

    def align_fov_mismatch(self, error_fname, regenerate=False, r_radia=(.5, .1), process_fn=None,
                           n_iter=(70, 50), crop_width=((160, 160), (32, 32))):
        # Align mismatch
        self.minus = MatlabRegister.align(self.minus, self.plus, cache_fname=error_fname, process_fn=process_fn,
                                          reproduce=regenerate, r_radia=r_radia, n_iter=n_iter)
        # remove redundancy edges
        self.minus = np.array(crop(self.minus, crop_width).array)
        self.plus = np.array(crop(self.plus, crop_width).array)

    @classmethod
    def processing(cls, raw_filename, background_fpath, error_fname=None, otsu_radii=50,
                   remove_strips=True, align_params=None, tie_params=None):
        bs_handle = cls(raw_filename)
        # Eliminate inhomogeneous illumination
        bs_handle.balance_illumination(background_fpath)
        # Image FoV registering
        _align_params = {} if not align_params else {k: v for k, v in align_params.items()}
        bs_handle.align_fov_mismatch(error_fname=error_fname, **_align_params)
        # Match Energy
        bs_handle.energy_matching()

        if remove_strips:  # Remove Strips
            lf_filter = partial(LowFrequencyFilter.interest_filtering, interest_radius=8)
            minus_corr, plus_corr = list(map(lf_filter, (bs_handle.minus, bs_handle.plus)))
            correction = (bs_handle.minus - minus_corr + invert(bs_handle.plus - plus_corr)) / 2
            correction_norm = correction - np.mean(correction)
            bs_handle.minus, bs_handle.plus = bs_handle.minus - correction_norm, bs_handle.plus + correction_norm

        _tie_params = {"wavelength": 532e-9, "pixel_size": 4.8e-6, "delta_d": 1e-3, "epsilon": 3e7}
        _tie_params.update(tie_params if tie_params else {})
        bs_handle.solve_phase(**_tie_params)
        bs_handle.phase_background_remove(otsu_radii=otsu_radii)
        return bs_handle.phase

    def data_save(self, focus_fname, phase_fname, minus_fname, plus_fname, modality_fname):
        multimodal_pairs = [(focus_fname, self.focus), (phase_fname, self.phase),
                            (minus_fname, self.minus), (plus_fname, self.plus)]
        for fname, arr in multimodal_pairs:
            i_min, i_max = np.min(arr), np.max(arr)
            if not (0. <= i_min <= 1. and 0. <= i_max <= 1.):
                arr = (arr - i_min) / (i_max - i_min)
            imsave(fname, img_as_ubyte(arr), "simpleitk")

        modalities = {"focus": img_as_ubyte(np.clip(self.focus, 0, 1.)),
                      "phase": self.phase.astype(np.float32),
                      "minus": img_as_ubyte(np.clip(self.minus, 0, 1.)),
                      "plus": img_as_ubyte(np.clip(self.plus, 0, 1.))}
        with open(modality_fname, "wb") as f:
            pickle.dump(modalities, f)

    @classmethod
    def load_collection(cls, path, suffix=".bmp", prefix="", fn_preprocess=None):
        for fname in listdir(path):
            if fname.startswith(prefix) and fname.endswith(suffix):
                yield cls(filename=join(path, fname), fn_preprocess=fn_preprocess)


class Transform:
    @staticmethod
    def process_truth_boxes(ground_truth_boxes, anchors, image_size, grid_size, n_classes, cls_location=0):
        """ Generate y (labels) of the training dataset from original labels for model training.
        The original labels has stored the measures of object bounding boxes as a list.
        However the list structure loose the spatial position, couldn't be used for yolo model
        training straightly. Therefore the labels should be scattered to array type which
        contain strong spatial relations for convolution operator.

        parameters:
            n_classes: decide the deeps of onehot

        inputs:
            label boxes - shape (N_labels, l-x1-y1-w1-h1)  (or x1-y1-w1-h1-l)
            the x1, y1, w1, h1 are the measures of bounding box in the input image plane, these measures
            could range from 0 to image_size (320) (for x1 and y1)

        outputs:
            detect_mask           [[GRID_SIZE, GRID_SIZE, N_anchor, 1],
            matching_gTruth_boxes  [GRID_SIZE, GRID_SIZE, N_anchor, x-y-w-h-l],
            class_onehot           [GRID_SIZE, GRID_SIZE, N_anchor, n_classes],
            gTruth_boxes_grid      [N_labels,  x-y-w-h-l]]
        """
        # ground_truth_boxes: [N_labels, 5]
        # Dictionary: (gTruth -> ground truth)
        scale = image_size // grid_size
        anchors = np.array(anchors).reshape((-1, 2))  # [N_anchor, 2]
        n_anchors = anchors.shape[0]

        # mask for object, for each grid, four boxes, one mask (box exist) value for each box
        detect_mask = np.zeros([grid_size, grid_size, n_anchors, 1])
        matching_ground_truth_boxes = np.zeros([grid_size, grid_size, n_anchors, 4 + 1])
        ground_truth_boxes_y = np.zeros_like(ground_truth_boxes)  # [N_labels, 5] => x1-y1-w1-h1-l

        for i, box in enumerate(ground_truth_boxes):  # [N_labels, l-x0-y0-w0-h0]
            # DB: tensor => numpy
            cls, coordinates = (box[0], box[1:]) if cls_location == 0 else (box[4], box[:4])
            x0, y0, w0, h0 = [elm / scale for elm in coordinates]
            ground_truth_boxes_y[i] = [x0, y0, w0, h0, cls]  # [N_labels,5] x0-y0-w0-h0-l

            if w0 * h0 > 0:  # valid box with object in it
                # Searching for best anchor according to IoU
                best_iou = 0
                best_anchor = 0
                for j in range(4):
                    interct = np.minimum(w0, anchors[j, 0]) * np.minimum(h0, anchors[j, 1])
                    union = w0 * h0 + (anchors[j, 0] * anchors[j, 1]) - interct
                    iou = interct / union

                    if iou > best_iou:  # best iou
                        best_iou = iou
                        best_anchor = j

                # Object exist inside this grid, change other y_label data
                if best_iou > 0:
                    x1 = np.floor(x0).astype(np.int32)
                    y1 = np.floor(y0).astype(np.int32)

                    detect_mask[y1, x1, best_anchor] = 1  # [b,h0,w0,4,1]
                    # [b, GRID_SIZE, GRID_SIZE, N_anchor, x0-y0-w0-h0-l]
                    matching_ground_truth_boxes[y1, x1, best_anchor] = np.array([x0, y0, w0, h0, cls])

        # Produce one-hot  (GRID_SIZE, GRID_SIZE, N_anchor, n_classes)
        onehot_base = np.expand_dims(matching_ground_truth_boxes[..., 4], axis=-1)
        class_onehot = np.concatenate([(onehot_base == j).astype(np.int8) for j in range(n_classes)], axis=-1)
        #  detect_mask           [[GRID_SIZE, GRID_SIZE, N_anchor, 1],
        #  matching_gTruth_boxes  [GRID_SIZE, GRID_SIZE, N_anchor, x-y-w-h-l],
        #  class_onehot           [GRID_SIZE, GRID_SIZE, N_anchor, n_classes],
        #  gTruth_boxes_grid      [N_labels,  x-y-w-h-l]]
        return detect_mask, matching_ground_truth_boxes, class_onehot, ground_truth_boxes_y

    @staticmethod
    def image_transform(image):
        # inputs are filenames, read it
        if isinstance(image, str):
            image = imread(image, True, "simpleitk")

        image = torch.from_numpy(image)
        mu = torch.mean(image, dim=(1, 2), keepdim=True)
        sigma = torch.std(image, dim=(1, 2), keepdim=True)
        image = torch.sigmoid((image - mu) / sigma)
        return image

    @staticmethod
    def target_transform(label: np.ndarray, cls_loc=0):
        """
        Parameters
        ---------
        label:
            [N_max, (cls, x, y, w, h)]
        cls_loc: int
            where is the class parameter is
        Returns
        -------
        detect_mask, gt_boxes, class_oh, box_grid """
        def to_tensor32(inputs): return [torch.from_numpy(x).to(F32) for x in inputs]
        return to_tensor32(Transform.process_truth_boxes(label, ANCHORS, DST_IMGSZ, GRIDSZ, N_CLASSES, cls_loc))


class BloodSmearDataset(Dataset):
    """ Loading the given files, produce a Dataset object. """
    def __init__(self, filepath="", training_stage="train", image_transform=None, target_transform=None,
                 load_modalities=("focus", "phase", "minus", "plus"), labels_folder="label",
                 ignores=None, dst_imgsz=DST_IMGSZ):
        self.max_boxes = 0
        self.root = filepath
        self.label_fnames = []
        self.modality_dict = {}

        self.training_stage = training_stage
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.destined_shape = (dst_imgsz, dst_imgsz)
        self.read_modalities = load_modalities
        self.labels_folder = labels_folder

        if exists(igfile := join(self.root, "ignored_filenames.txt")):
            ignores = igfile
        if ignores is not None:
            self.ignore_filenames = []
            if isinstance(ignores, str):
                with open(ignores, "r") as f:
                    for line in f.readlines():
                        self.ignore_filenames.append(line.rstrip("\n"))
            else:
                self.ignore_filenames = ignores
        else:
            self.ignore_filenames = []
        self.dataset_scan()

    @staticmethod
    def count_max_bboxes(counter_filename):
        def _count_single_file(_filename):
            with open(_filename, "r") as _f:
                data = _f.readlines()
            return len(data)

        length = []
        labels_folder = dirname(counter_filename)
        for stage in ("train", "valid", "test"):
            stage_max = 0
            fullpath = join(labels_folder, stage)
            for file in listdir(fullpath):
                if (file_max := _count_single_file(join(fullpath, file))) > stage_max:
                    stage_max = file_max
            length.append(stage_max)

        with open(counter_filename, "w") as f:
            yaml.dump({"length": length}, f)

    def dataset_scan(self):
        """ Prepare for __getitem__ """
        # loads all the file names will be used
        self.modality_dict = {mod: [f for f in listdir(join(self.root, mod, self.training_stage))
                                    if f not in self.ignore_filenames]
                              for mod in self.read_modalities}
        self.label_fnames = [f for f in listdir(join(self.root, self.labels_folder, self.training_stage))
                             if f not in self.ignore_filenames]
        max_bboxes_file = join(self.root, self.labels_folder, "max_bboxes.yaml")
        if not exists(max_bboxes_file):
            self.count_max_bboxes(max_bboxes_file)
        with open(max_bboxes_file, "r") as f:
            length = yaml.load(f, yaml.FullLoader)
        self.max_boxes = length.get("length")[("train", "valid", "test").index(self.training_stage)]

    def labels_loader(self, filename):
        boxes = []
        dh, dw = self.destined_shape
        with open(filename, "r") as _f:
            for line in _f:
                # processing one line to a box data, the x-y-w-h in here is normalized to 0~1
                cls, x, y, w, h = [float(elm) for elm in line.rstrip("\n").split(" ")]
                boxes.append((cls, x * dw, y * dh, w * dw, h * dh))

        # Packaging the labels using uniform size array: [N_samples, n_label_per_sample, 5]
        boxes_array = np.zeros((self.max_boxes, 5))
        boxes_array[:len(boxes)] = np.asarray(boxes)  # overwrite the N boxes_array info  [N,5]
        return boxes_array

    def modality_loader(self, filenames):
        def _validate_same_sample(_filenames: List[str]):
            # '20210902-10000.jpg'   r'\d{8}\-\d{5}.\w{3}'
            file_ids = []
            for name in _filenames:
                assert re.match(r'\d{8}-\d{5}.\w{3}', name)
                file_ids.append(name.split(".")[0])
            try:
                assert all([fid == file_ids[0] for fid in file_ids])
                return True
            except AssertionError:
                return False

        # mod_filenames - {list: 4} - [amp_fname, pha_fname, minus_fname, plus_fname]
        assert _validate_same_sample([basename(fname) for fname in filenames])
        # Read from disk
        modalities = [img_as_float32(imread(fname, True, "simpleitk")) for fname in filenames]
        # Resize to destined image shape
        modalities = [resize(mod, self.destined_shape) for mod in modalities]
        # Aggregates to single ndarray OR expand dimension of single channel inputs
        return np.stack(modalities)

    def __len__(self):
        return len(self.label_fnames)

    def get_item_names(self, index):
        label_name = join(self.root, self.labels_folder, self.training_stage, self.label_fnames[index])
        modality_names = [join(self.root, mod, self.training_stage, self.modality_dict.get(mod)[index])
                          for mod in self.read_modalities]
        return label_name, modality_names

    def __getitem__(self, idx):
        label_name, modality_names = self.get_item_names(idx)
        labels = self.labels_loader(label_name)
        modalities = self.modality_loader(modality_names)
        # img numpy(320,320) label list(295,5) -> img (320,320) label (295,5)
        # the shape of input images and labels might not corresponding to the image shape of (320, 320)
        if self.image_transform:
            modalities = self.image_transform(modalities)
        if self.target_transform:
            labels = self.target_transform(labels)
        return modalities, labels


class NameRememberedDataset(BloodSmearDataset):
    def __init__(self, **kwargs): super(NameRememberedDataset, self).__init__(**kwargs)

    def __getitem__(self, idx):
        label_name, modality_names = self.get_item_names(idx)
        return (self.image_transform(self.modality_loader(modality_names)),
                self.target_transform(self.labels_loader(label_name)),
                (label_name, modality_names))


def create_dataloader(fpath_str: str, training_stage, batch_size, image_tf=None, target_tf=None,
                      dataset_obj=BloodSmearDataset, load_modalities=("focus", "phase", "minus", "plus"),
                      **loader_params):
    """ Dataloader constructor """
    constructor = {"filepath": fpath_str,
                   "training_stage": training_stage,
                   "load_modalities": load_modalities,
                   "image_transform": Transform.image_transform if image_tf is None else image_tf,
                   "target_transform": Transform.target_transform if target_tf is None else target_tf}
    return DataLoader(batch_size=batch_size, dataset=dataset_obj(**constructor), **loader_params)


if __name__ == "__main__":
    # Logging Config -----------------------------------------------------------------
    logging.config.fileConfig("..\\log\\config\\dataset.conf")
    logger = logging.getLogger(__name__)
    # Logging Config Ended -----------------------------------------------------------
    try:
        temp_train_load = create_dataloader(SET202109_FILENAME, "train", 4)
        pass
    except Exception as err:
        logger.exception(err)
