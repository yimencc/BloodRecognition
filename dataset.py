import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import shutil
from typing import List
from os.path import join
from xml.etree import ElementTree as ET

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import figure
from skimage.io import imread
from torch.utils.data import Dataset

from cccode.image import Check

IMGSZ       =   320
GRIDSZ      =   40
IMG_PLUGIN  =   "simpleitk"
F32         =   torch.float32
ck          =   Check(False, False, False)
ANCHORS     =   [1., 1., 1.125, 1.125, 1.25, 1.25, 1.375, 1.375]
FFOV_XML    =   "D:\\Workspace\\RBC Recognition\\data\\2021-01-05\\fov_annotations.xml"


class DataTransform:
    @staticmethod
    def process_true_boxes(gt_boxes, anchors, image_size):
        # gt_boxes: [296, 5]
        # 320 // 40 = 8
        scale = image_size // GRIDSZ
        # [4, 2]
        anchors = np.array(anchors).reshape((4, 2))

        # mask for object, for each grid, four boxes
        # one mask (box exist) value for each box
        detector_mask = np.zeros([GRIDSZ, GRIDSZ, 4, 1])
        # for each grid, four boxes
        # five value for each box: x-y-w-h-l
        matching_gt_box = np.zeros([GRIDSZ, GRIDSZ, 4, 5])
        # [40,5] x1-y1-x2-y2-l => x-y-w-h-l
        gt_boxes_grid = np.zeros(gt_boxes.shape)

        for i, box in enumerate(gt_boxes):  # [286,5]
            # DB: tensor => numpy
            # box: [5], x1-y1-x2-y2-l
            # 320 => 32
            x = box[0] / scale
            y = box[1] / scale
            w = box[2] / scale
            h = box[3] / scale
            # [286,5] x-y-w-h-l
            gt_boxes_grid[i] = np.array([x, y, w, h, box[4]])

            if w * h > 0:  # valid box with object in it
                # Searching for best anchor according to IoU
                best_anchor = 0
                best_iou = 0
                for j in range(4):
                    interct = np.minimum(w, anchors[j, 0])*np.minimum(h, anchors[j, 1])
                    union = w * h + (anchors[j, 0] * anchors[j, 1]) - interct
                    iou = interct / union

                    if iou > best_iou:  # best iou
                        best_anchor = j
                        best_iou = iou
                        # found the best anchors
                if best_iou > 0:
                    x_coord = np.floor(x).astype(np.int32)
                    y_coord = np.floor(y).astype(np.int32)
                    # [b,h,w,4,1]
                    detector_mask[y_coord, x_coord, best_anchor] = 1
                    # [b,h,w,4,x-y-w-h-l]
                    matching_gt_box[y_coord, x_coord, best_anchor] = np.array([x, y, w, h, box[4]])

        # [296,5] => [32,32,4,5]
        # [32,32,4,5]
        # [32,32,4,1]
        # [296,5]
        return matching_gt_box, detector_mask, gt_boxes_grid

    @staticmethod
    def image_transform(image):
        image = torch.tensor(image)
        mu = torch.mean(image, dim=(1, 2), keepdim=True)
        sigma = torch.std(image, dim=(1, 2), keepdim=True)
        image = torch.sigmoid((image-mu)/sigma)
        return image

    @classmethod
    def target_transform(cls, label) -> tuple:
        """
        Returns
        -------
        mask, gt_box, class_oh, box_grid
        """
        gt_box, mask, grid = map(lambda x: torch.from_numpy(x).to(F32),
                                 cls.process_true_boxes(label, ANCHORS, IMGSZ))
        oh_base = torch.tile(torch.zeros_like(mask), (1, 1, 1, 5))
        class_oh = oh_base.scatter_(-1, gt_box[..., 4:].to(torch.int64), 1)
        return mask, gt_box, class_oh[..., 1:], grid


class MultimodalSample:
    def __init__(self):
        self.sample_idx = None
        self.phase      = None
        self.amplitude  = None
        self.overfocus  = None
        self.underfocus = None
        self.labels     = []

    @property
    def modalities(self):
        return self.amplitude, self.phase, self.underfocus, self.overfocus

    def set_modalities(self, amp, pha, under, over):
        self.amplitude  = amp
        self.phase      = pha
        self.overfocus  = over
        self.underfocus = under

    @classmethod
    def from_dict(cls, dictionary):
        sample = cls()
        sample.sample_idx = dictionary["image_idx"]
        sample.labels     = dictionary["labels"]
        sample.phase      = imread(dictionary["pha_fullname"])
        sample.amplitude  = imread(dictionary["amp_fullname"])
        sample.overfocus  = imread(dictionary["over_fullname"])
        sample.underfocus = imread(dictionary["under_fullname"])
        return sample

    def label_reshape(self):
        labels_array = []
        for cls, bbox in self.labels:
            labels_array.append([*bbox, cls])
        return np.array(labels_array)

    labels_array = property(label_reshape)

    def annotate_axes(self, ax: mpl.figure.Axes):
        for _, bbox in self.labels:
            x, y, w, h = bbox
            rect = plt.Rectangle((x-w//2-1, y-h//2-1), w, h, fill=False, color="blue")
            ax.add_patch(rect)
        return ax


class AnnotationParser:
    AmpNameStr   = "amp_fullname"
    PhaNameStr   = "pha_fullname"
    OverNameStr  = "over_fullname"
    UnderNameStr = "under_fullname"
    NameStrings  = [AmpNameStr, PhaNameStr, OverNameStr, UnderNameStr]

    def __init__(self, filename):
        etree = ET.parse(filename)
        self.elementsRoot: ET.Element           = etree.getroot()
        self.sampleElements: list               = self.elementsRoot.findall("sample")
        self.samples: List[MultimodalSample]    = [MultimodalSample.from_dict(self._parse_element_to_dict(sp))
                                                   for sp in self.sampleElements]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def _parse_element_to_dict(self, element: ET.Element) -> dict:
        sample_dict = {"image_idx": element.find("image_idx").text}

        for name_string in self.NameStrings:
            sample_dict.update({name_string: element.find(name_string).text})

        # Extracting all label bboxes
        labels      =   []
        labels_root =   element.find("labels")
        for label_element in labels_root.findall("label"):
            cls     =   int(label_element.find("class").text)
            bbox    =   [int(label_element.find("bbox").find(tag).text) for tag in ["x", "y", "w", "h"]]
            labels.append((cls, bbox))
        sample_dict.update({"labels": labels})
        return sample_dict

    @staticmethod
    def split_sample(sample: MultimodalSample, n_split=3, target_size=(320, 320)) -> List[MultimodalSample]:
        assert (sample.phase.shape == sample.amplitude.shape ==
                sample.underfocus.shape == sample.overfocus.shape)

        height, width               =   sample.phase.shape
        tgt_height, tgt_width       =   target_size
        centroid_arrange_width      =   width-tgt_width
        centroid_arrange_height     =   height-tgt_height

        centroid_interval_width     =   centroid_arrange_width  // n_split
        centroid_interval_height    =   centroid_arrange_height // n_split

        subview_multimodal_samples  =   []
        for i in range(n_split):        # ROW split
            for k in range(n_split):    # COLUMN split
                x_centroid = tgt_width//2 + k*centroid_interval_width
                y_centroid = tgt_height//2 + i*centroid_interval_height

                x0 = x_centroid - tgt_width//2
                x1 = x_centroid + tgt_width//2
                y0 = y_centroid - tgt_height//2
                y1 = y_centroid + tgt_height//2

                subview_modalities = [image[y0:y1, x0:x1] for image in sample.modalities]

                subview_labels  = []
                for cls, [x, y, w, h] in sample.labels:
                    if x0 <= x < x1 and y0 <= y < y1:
                        subview_labels.append((cls, [x-x0, y-y0, w, h]))

                subview_mltSample = MultimodalSample()
                subview_mltSample.labels  = subview_labels
                subview_mltSample.set_modalities(*subview_modalities)
                subview_multimodal_samples.append(subview_mltSample)
        return subview_multimodal_samples

    def subview_dataset(self, set_type="training") -> np.ndarray:
        set_length  =   len(self.samples)
        train_len   =   int(0.8 * set_length)
        valid_len   =   int(0.9 * set_length)

        if set_type == "training":
            return np.hstack([self.split_sample(sp) for sp in self.samples[:train_len]])
        elif set_type == "validating":
            return np.hstack([self.split_sample(sp) for sp in self.samples[train_len:valid_len]])
        else:  # set_type == "testing"
            return np.hstack([self.split_sample(sp) for sp in self.samples[valid_len:]])


class RBCXmlDataset(Dataset):
    def __init__(self, xml_filename, set_name, image_transform=None, target_transform=None):
        self.max_boxes                  =   0
        self.image_transform            =   image_transform
        self.target_transform           =   target_transform

        parser                          =   AnnotationParser(xml_filename)
        sample_sets                     =   parser.subview_dataset(set_name)
        self.modalities, self.labels    =   self.load_datasets(sample_sets)

    def load_datasets(self, sample_sets: np.ndarray):
        # samplesets: np.array([MltSample1, MltSample2, MltSample3, ....])
        labels     = []
        modalities = []
        for i, subview_sample in enumerate(sample_sets):
            modalities.append(np.array(subview_sample.modalities))

            labels_array = subview_sample.labels_array
            if len(labels_array) > self.max_boxes:
                self.max_boxes = len(labels_array)
            labels.append(labels_array)

        boxes = np.zeros((len(labels), self.max_boxes, 5))
        for i, label in enumerate(labels):
            # overwrite the N boxes info  [N,5]
            boxes[i, :label.shape[0]] = label

        return modalities, boxes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label       =   self.labels[idx]
        modality    =   self.modalities[idx]
        # img numpy(320,320) label list(295,5) -> img (320,320) label (295,5)
        if self.image_transform:
            modality = self.image_transform(modality)
        if self.target_transform:
            label   = self.target_transform(label)
        sample = {"modality": modality, "label": label}
        return sample


TRAIN_DS_CONSTRUCTOR = {"xml_filename":     FFOV_XML,
                        "set_name":         "training",
                        "image_transform":  DataTransform.image_transform,
                        "target_transform": DataTransform.target_transform}

VALID_DS_CONSTRUCTOR = {"xml_filename":     FFOV_XML,
                        "set_name":         "validating",
                        "image_transform":  DataTransform.image_transform,
                        "target_transform": DataTransform.target_transform}


def AnnotationParser_test():
    trainRBCDataset =   RBCXmlDataset(**TRAIN_DS_CONSTRUCTOR)
    validRBCDataset =   RBCXmlDataset(**VALID_DS_CONSTRUCTOR)


if __name__ == '__main__':
    AnnotationParser_test()
