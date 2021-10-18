import os
import re
import pickle
import logging.config
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from functools              import partial
from typing                 import List, Optional
from pprint                 import pformat
from os.path                import join, split, isfile, isdir
from xml.etree              import ElementTree as ET
from xml.dom.minidom        import parseString

import torch
import numpy                as np
import matplotlib           as mpl
import matplotlib.pyplot    as plt
from cv2                    import resize
from matplotlib             import figure
from skimage.io             import imread, imsave
from skimage.util           import dtype_limits, img_as_ubyte, img_as_float32
from torch.utils.data       import Dataset, DataLoader

from cccode.image                   import Check
from Deeplearning.util.functions    import single_sample_visualization

DST_IMGSZ   =   320
SRC_IMGSZ   =   340
GRIDSZ      =   40
N_CLASSES   =   3
IMG_PLUGIN  =   "simpleitk"
F32         =   torch.float32
ck          =   Check(False, False, False)
ANCHORS     =   [1., 1., 1.125, 1.125, 1.25, 1.25, 1.375, 1.375]
DATA_ROOT   =   "D:\\Workspace\\RBC Recognition\\datasets"


class MultimodalSample:
    """
    labels: [(cls, x, y, w, h)]
    """
    def __init__(self):
        self.sample_idx = None
        self.phase      = None
        self.amplitude  = None
        self.overfocus  = None
        self.underfocus = None
        self.labels     = []
        self.source_msg = {}
        self.image_shape = None

    def __repr__(self):
        output      =  f"<Ojb MultimodalSample>\n"
        if self.sample_idx is not None:
            output  +=  f"\timage_index:    {self.sample_idx:<4d}\n"

        names = ["amplitude", "phase", "under-focus", "over-focus"]
        str_max_len = np.max([len(n) for n in names]) + 3
        for modality, name in zip(self.modalities, names):
            output += "\t" + "{name}:".ljust(str_max_len) + f"{type(modality)}, {modality.shape.__repr__()}," \
                                                            f"val_range: {dtype_limits(modality)}\n"
        output      +=  f"\tlabel numbers:  {len(self.labels)}\n\tsource_messages:\n"
        output      +=  pformat(self.source_msg, indent=8)
        return output

    @property
    def modalities(self):
        return self.amplitude, self.phase, self.underfocus, self.overfocus

    def set_modalities(self, amp, pha, under, over):
        self.amplitude, self.phase, self.underfocus, self.overfocus = amp, pha, under, over

    @classmethod
    def from_element(cls, sample: ET.Element, read_image=True):
        mm_sample       =   cls()
        amp_fullname    =   sample.find("amp_fullname").text
        pha_fullname    =   sample.find("pha_fullname").text
        over_fullname   =   sample.find("over_fullname").text
        under_fullname  =   sample.find("under_fullname").text

        mm_sample.source_msg.update({"amp_fullname": amp_fullname, "pha_fullname": pha_fullname,
                                     "over_fullname": over_fullname, "under_fullname": under_fullname})

        mm_sample.sample_idx =   int(sample.find("image_idx").text)

        if read_image:
            mm_sample.amplitude  =   imread(amp_fullname,    True, "simpleitk")
            mm_sample.phase      =   imread(pha_fullname,    True, "simpleitk")
            mm_sample.overfocus  =   imread(over_fullname,   True, "simpleitk")
            mm_sample.underfocus =   imread(under_fullname,  True, "simpleitk")

        img_shape_elm   =   sample.find("image_shape")
        if img_shape_elm is not None:
            height, width   =   [int(img_shape_elm.find(tag).text) for tag in ["height", "width"]]
            mm_sample.image_shape = (height, width)
            if mm_sample.amplitude is not None:
                assert (height, width) == mm_sample.amplitude.shape

        labels_elm      =   sample.find("labels")
        for lbl_elm in labels_elm.findall("label"):
            bbox_elm    =   lbl_elm.find("bbox")
            x, y, w, h  =   [int(bbox_elm.find(tag).text) for tag in ("x", "y", "w", "h")]
            lbl_cls     =   int(lbl_elm.find("class").text)
            mm_sample.labels.append((lbl_cls, x, y, w, h))
        return mm_sample

    @staticmethod
    def split(sample, n_split=3, target_size=(320, 320)):
        assert (sample.phase.shape == sample.amplitude.shape ==
                sample.underfocus.shape == sample.overfocus.shape)

        height,     width           =   sample.phase.shape
        tgt_height, tgt_width       =   target_size
        centroid_arrange_width      =   width - tgt_width
        centroid_arrange_height     =   height - tgt_height

        centroid_interval_width     =   centroid_arrange_width  // (n_split-1)
        centroid_interval_height    =   centroid_arrange_height // (n_split-1)

        subview_multimodal_samples: List[MultimodalSample]  =   []
        for i in range(n_split):        # ROW split
            for k in range(n_split):    # COLUMN split
                x_centroid = tgt_width//2  + k*centroid_interval_width
                y_centroid = tgt_height//2 + i*centroid_interval_height

                x0 = x_centroid - tgt_width//2
                x1 = x_centroid + tgt_width//2
                y0 = y_centroid - tgt_height//2
                y1 = y_centroid + tgt_height//2

                subview_modalities = [image[y0:y1, x0:x1] for image in sample.modalities]

                subview_labels  = []
                for cls, x, y, w, h in sample.labels:
                    if x0 <= x < x1 and y0 <= y < y1:
                        subview_labels.append((cls, x-x0, y-y0, w, h))

                subview_mltSample = MultimodalSample()
                subview_mltSample.labels  = subview_labels
                subview_mltSample.source_msg.update({"parent_msg": sample.source_msg})
                subview_mltSample.set_modalities(*subview_modalities)
                subview_multimodal_samples.append(subview_mltSample)
        return subview_multimodal_samples

    def save(self, save_root):
        amp_path, pha_path, minus_path, plus_path, anno_path = [join(save_root, pth) for pth in
                                                                ("amp", "pha", "minus", "plus", "anno")]
        for pth in (amp_path, pha_path, minus_path, plus_path, anno_path):
            if not os.path.exists(pth):
                os.mkdir(pth)

        amp_fullname        =   join(amp_path,      f"amp_{self.sample_idx:04d}.jpg")
        pha_fullname        =   join(pha_path,      f"pha_{self.sample_idx:04d}.jpg")
        minus_fullname      =   join(minus_path,    f"minus_{self.sample_idx:04d}.jpg")
        plus_fullname       =   join(plus_path,     f"plus_{self.sample_idx:04d}.jpg")
        anno_fullname       =   join(anno_path,     f"pha_{self.sample_idx:04d}.txt")

        # save images
        for modality, fname in zip(self.modalities, (amp_fullname, pha_fullname, minus_fullname, plus_fullname)):
            # amplitude, phase, underfocus, overfocus
            imsave(fname, img_as_ubyte(modality), "simpleitk")

        # save labels
        yolo_string     =   ""
        height, width   =   self.amplitude.shape
        for cell_class, x, y, w, h in self.labels:
            # Coordinate transform from absolute coordination to yolo
            x1, y1 = x / width, y / height
            w1, h1 = w / width, h / height
            coord_str = " ".join([f"{coord:1.6f}" for coord in (x1, y1, w1, h1)])
            yolo_string += str(cell_class) + " " + coord_str + "\n"

        # save single sample yolo file
        with open(anno_fullname, "w") as f:
            f.write(yolo_string)

    def out_labeled_data(self, dst_image_shape: tuple[int],
                         modalities_to: Optional[list] = None,
                         labels_to: Optional[list] = None):
        # cast image and labels to destined shape
        if self.image_shape is None:
            self.image_shape = self.modalities[0].shape

        if self.image_shape != dst_image_shape:
            self.set_modalities(*[resize(img, dst_image_shape) for img in self.modalities])
            for i, lbl in enumerate(self.labels):
                scale_x = dst_image_shape[1]/self.image_shape[1]
                scale_y = dst_image_shape[0]/self.image_shape[0]
                cls, x, y, w, h = lbl
                x, w = x*scale_x, w*scale_x
                y, h = y*scale_x, h*scale_y
                self.labels[i] = cls, x, y, w, h
        if modalities_to and labels_to:
            modalities_to.append(np.array(self.modalities))
            labels_to.append(np.array(self.labels))
        else:
            return np.array(self.modalities), np.array(self.labels)

    def annotate_axes(self, ax: mpl.figure.Axes):
        for _, x, y, w, h in self.labels:
            rect = plt.Rectangle((x-w//2-1, y-h//2-1), w, h, fill=False, color="blue")
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
    def fromXML(cls, filename):
        container: StandardXMLContainer     =   cls()
        etree                               =   ET.parse(filename)
        container.root                      =   etree.getroot()
        container.sample_elements           =   container.root.findall("sample")
        container.src_filename              =   filename
        return container

    def sample_collection_visualization(self, start_idx=0, figsize=(20, 12), title=None, tt_fontsize=16):
        visualized_samples = [MultimodalSample.from_element(sp) for sp in
                              self.sample_elements[start_idx:start_idx + 15]]
        fig, axs = plt.subplots(3, 5, figsize=figsize, constrained_layout=True)
        if title is not None:
            plt.suptitle(title, fontproperties={"size": tt_fontsize})
        for row in range(3):
            for collum in range(5):
                n_sp =   row*5 + collum
                ax   =   axs[row, collum]
                sample: MultimodalSample = visualized_samples[n_sp]
                ax.imshow(sample.phase, cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ("top", "bottom", "left", "right"):
                    ax.spines[sp].set_visible(False)
                sample.annotate_axes(ax)
        plt.show()

    @staticmethod
    def subElem(parent: ET.Element, child_tag: str, text: str = None):
        child = ET.SubElement(parent, child_tag)
        if text is not None:
            child.text = text if isinstance(text, str) else str(text)
        return child

    @staticmethod
    def sample_msg(sample: ET.Element):
        img_idx_str     =   sample.find('image_idx').text
        height, width   =   [int(sample.find("image_shape").find(tag).text) for tag in ("height", "width")]
        output = f"Sample\n\timage idx:      {img_idx_str}\n\timage shape:    ({height}, {height})\n"

        amp_fpath,      amp_fname   =   split(sample.find("amp_fullname").text)
        pha_fpath,      pha_fname   =   split(sample.find("pha_fullname").text)
        over_fpath,     over_fname  =   split(sample.find("over_fullname").text)
        under_fpath,    under_fname =   split(sample.find("under_fullname").text)
        amp_root,       amp_dir     =   split(amp_fpath)
        pha_root,       pha_dir     =   split(pha_fpath)
        over_root,      over_dir    =   split(over_fpath)
        under_root,     under_dir   =   split(under_fpath)
        try:
            assert amp_root == pha_root == over_root == under_root
        except AssertionError as e:
            logger.exception(e)

        num_labels  =   len(sample.find("labels").findall("label"))
        output      +=  f"\tmodalities root: {amp_root}\n"
        output      +=  f"\t\tamplitude     dir: {amp_dir:<6s} filename: {amp_fname}\n"
        output      +=  f"\t\tphase         dir: {pha_dir:<6s} filename: {pha_fname}\n"
        output      +=  f"\t\tunder-focus   dir: {over_dir:<6s} filename: {under_fname}\n"
        output      +=  f"\t\tover-focus    dir: {under_dir:<6s} filename: {over_fname}\n"
        output      +=  f"\tlabels number:  {num_labels}\n"
        return output

    def add_sample(self, idx, image_shape, amplitude_filename, phase_filename, minus_filename, plus_filename):
        sample  =   ET.SubElement(self.root, "sample")
        shape   =   self.subElem(sample, "image_shape")
        self.subElem(shape,  "height",          image_shape[0])
        self.subElem(shape,  "width",           image_shape[1])

        self.subElem(sample, "image_idx",       str(idx))
        self.subElem(sample, "amp_fullname",    amplitude_filename)
        self.subElem(sample, "pha_fullname",    phase_filename)
        self.subElem(sample, "over_fullname",   plus_filename)
        self.subElem(sample, "under_fullname",  minus_filename)
        self.subElem(sample, "labels")
        return sample

    def add_label(self, sample: ET.Element, x, y, w, h, cell_class, creator: str):
        node  = sample.find("labels")
        label = ET.SubElement(node,  "label")
        bbox  = ET.SubElement(label, "bbox")
        self.subElem(label, "class",    str(cell_class))
        self.subElem(label, "creator",  creator)

        # add all bounding box values
        assert isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int)
        self.subElem(bbox, "x", int(x))
        self.subElem(bbox, "y", int(y))
        self.subElem(bbox, "w", int(w))
        self.subElem(bbox, "h", int(h))

    def compile(self, fpath: str = None, prettify=True):
        xml_string = ET.tostring(self.root, encoding="utf-8")
        if prettify:
            reparsed    =   parseString(xml_string)
            xml_string  =   reparsed.toprettyxml(indent="\t")

        with open(fpath, "w") as f:
            f.write(xml_string)

    def toyolo(self, yolopath, prefix="phase"):
        yolo_pattern    =   "phase_{:02d}.txt" if prefix == "phase" else "img_{:02d}.txt"

        for sample in self.root.findall("sample"):
            img_idx         =   int(sample.find("image_idx").text)
            image_shape     =   sample.find("image_shape")
            height, width   =   [int(image_shape.find(tag).text) for tag in ("height", "width")]

            yolo_string     =   ""
            for lbl in sample.find("labels").findall("label"):
                bbox        =   lbl.find("bbox")
                x, y, w, h  =   [int(bbox.find(tag).text) for tag in ("x", "y", "w", "h")]
                cell_class  =   lbl.find("class").text

                # Coordinate transform from absolute coordination to yolo
                x1, y1      =   x / width, y / height
                w1, h1      =   w / width, h / height
                coord_str   =   " ".join([f"{coord:1.6f}" for coord in (x1, y1, w1, h1)])
                yolo_string +=  cell_class + " " + coord_str + "\n"

            # save single sample yolo file
            dst_fullname    =   join(yolopath, yolo_pattern.format(img_idx))
            with open(dst_fullname, "w") as f:
                f.write(yolo_string)

    def sample_slicing(self, split_root="", sample_per_batch=10, dst_shape=(340, 340)) -> List[MultimodalSample]:
        """ Slicing all samples into subview samples """
        # TODO: dst_shape should determined through dataset configuration
        batch_path  =   []
        sample_set  =   []
        n_samples   =   len(self.sample_elements)

        if split_root != "":    # Preparing for storage
            batch_nums  =   n_samples // sample_per_batch + 1
            batch_path  =   [join(split_root, f"batch_{n_bth:02d}") for n_bth in range(batch_nums)]
            for pth in [split_root] + batch_path:
                if not os.path.exists(pth):
                    os.mkdir(pth)

        for i, sp in enumerate(self.sample_elements):
            sp          =   MultimodalSample.from_element(sp)   # XML <Elements> to <MultimodalSample>
            children    =   MultimodalSample.split(sp, target_size=dst_shape)
            for k, child in enumerate(children):
                child.sample_idx = i*9 + k
                if split_root:   # Perform storage
                    save_root = batch_path[i // sample_per_batch] if sample_per_batch  else split_root
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
    xml_container       =   StandardXMLContainer()
    for *fnames, ann_fname in zip(focus_filenames, phase_filenames, minus_filenames, plus_filenames, ann_filenames):
        indexes     =   [fname.rstrip(imgfmt).lstrip(pref) for fname, pref in
                         zip(fnames, ("focus_", "phase_", "minus_", "plus_"))]
        ann_idx     =   ann_fname.removesuffix(".txt").removeprefix("auto_")

        try:
            assert all([ann_idx == idx for idx in indexes])
        except AssertionError as e:
            logger.exception(e)

        fullnames       =   [join(ph, fn) for ph, fn in zip((focus_path, phase_path, minus_path, plus_path), fnames)]
        ann_fullname    =   join(annotations_path,  ann_fname)

        # add sample
        img_idx         =   int(ann_idx)
        image_shape     =   imread(fullnames[0], True, "simpleitk").shape  # load image shape
        sample          =   xml_container.add_sample(img_idx, image_shape, *fullnames)
        # Loading the annotations
        with open(ann_fullname, "r") as f:
            for line in f:
                if line.endswith("\n"):
                    line = line.rstrip("\n")
                num_strings     =   list(filter(lambda elm: elm != "", line.split(" ")))
                x, y, w, h      =   [int(num) for num in num_strings]

                if fixed_class == "rbc":    # Add label
                    clsn = 0
                xml_container.add_label(sample, x, y, w, h, clsn, creator)

    # writes into disk
    xml_container.compile(xml_filename)
    return xml_container


def read_yolo_labels_set(filenames: str, dst_shape):
    _H, _W = dst_shape

    def _read_single_label(fname):
        boxes = []
        with open(fname, "r") as _f:
            for line in _f:
                # processing one line to a box data, the x-y-w-h in here is normalized to 0~1
                cls, x, y, w, h = [float(elm) for elm in line.rstrip("\n").split(" ")]
                x, w = x*_W, w*_W
                y, h = y*_H, h*_H
                boxes.append((cls, x, y, w, h))
        return boxes

    label_set = [_read_single_label(f) for f in filenames]
    max_boxes = max((len(lbl) for lbl in label_set))

    # Packaging the labels using uniform size array: [N_samples, n_label_per_sample, 5]
    boxes_set = np.zeros((len(label_set), max_boxes, 5))
    for i, label in enumerate(label_set):  # overwrite the N boxes_set info  [N,5]
        boxes_set[i, :len(label)] = label
    return boxes_set


def validate_same_sample(filenames: List[str]):
    # '20210902-10000.jpg'   r'\d{8}\-\d{5}.\w{3}'
    file_ids = []
    for name in filenames:
        assert re.match(r'\d{8}-\d{5}.\w{3}', name)
        file_ids.append(name.split(".")[0])
    try:
        assert all([fid == file_ids[0] for fid in file_ids])
        return True
    except AssertionError:
        return False


def load_modalities_training_data(src_fpath: str, training_stage: str, dst_imsz):
    """ Read all the filename in the target path """
    sub_directories = ("amp", "pha", "minus", "plus", "anno")
    for subdir in sub_directories:
        assert os.path.isdir(join(src_fpath, subdir, training_stage))

    # {'anno': ['pha_0000.txt', ...], 'amp': ['amp_0000.jpg', ...], ...}
    dataset_dict = {subdir: [join(src_fpath, subdir, training_stage, f)
                             for f in os.listdir(join(src_fpath, subdir, training_stage))]
                    for subdir in sub_directories}

    # read all labels into a single numpy ndarray
    label_set = read_yolo_labels_set(dataset_dict["anno"], (dst_imsz, dst_imsz))

    modalities_set = []     # Load Modalities set
    for mod_filenames in zip(*[dataset_dict[mod] for mod in ("amp", "pha", "minus", "plus")]):
        assert validate_same_sample([os.path.split(fname)[-1] for fname in mod_filenames])
        modalities = [img_as_float32(imread(fname, True, "simpleitk")) for fname in mod_filenames]
        modalities = np.vstack([resize(mod, (dst_imsz, dst_imsz))[np.newaxis, :] for mod in modalities])
        modalities_set.append(modalities)
    return modalities_set, label_set


class DataTransform:
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
        TODO: Arguments should be determined by the caller
        """
        # ground_truth_boxes: [N_labels, 5]
        # Dictionary: (gTruth -> ground truth)
        scale       =   image_size // grid_size
        anchors     =   np.array(anchors).reshape((-1, 2))    # [N_anchor, 2]
        n_anchors   =   anchors.shape[0]

        # mask for object, for each grid, four boxes, one mask (box exist) value for each box
        detect_mask             =   np.zeros([grid_size, grid_size, n_anchors, 1])
        matching_gTruth_boxes   =   np.zeros([grid_size, grid_size, n_anchors, 4+1])
        gTruth_boxes_grid       =   np.zeros_like(ground_truth_boxes)     # [N_labels, 5] => x1-y1-w1-h1-l

        for i, box in enumerate(ground_truth_boxes):  # [N_labels, l-x0-y0-w0-h0]
            # DB: tensor => numpy
            cls, coordinates        =   (box[0], box[1:]) if cls_location == 0 else (box[4], box[:4])
            x0, y0, w0, h0          =   [elm / scale for elm in coordinates]
            gTruth_boxes_grid[i]    =   [x0, y0, w0, h0, cls]                       # [N_labels,5] x0-y0-w0-h0-l

            if w0 * h0 > 0:  # valid box with object in it
                # Searching for best anchor according to IoU
                best_iou    =   0
                best_anchor =   0
                for j in range(4):
                    interct =   np.minimum(w0, anchors[j, 0])*np.minimum(h0, anchors[j, 1])
                    union   =   w0 * h0 + (anchors[j, 0] * anchors[j, 1]) - interct
                    iou     =   interct / union

                    if iou > best_iou:  # best iou
                        best_iou    =   iou
                        best_anchor =   j

                # Object exist inside this grid, change other y_label data
                if best_iou > 0:
                    x1 = np.floor(x0).astype(np.int32)
                    y1 = np.floor(y0).astype(np.int32)

                    detect_mask[y1, x1, best_anchor] = 1    # [b,h0,w0,4,1]
                    # [b, GRID_SIZE, GRID_SIZE, N_anchor, x0-y0-w0-h0-l]
                    matching_gTruth_boxes[y1, x1, best_anchor] = np.array([x0, y0, w0, h0, cls])

        # Produce one-hot  (GRID_SIZE, GRID_SIZE, N_anchor, n_classes)
        onehot_base = np.expand_dims(matching_gTruth_boxes[..., 4], axis=-1)
        class_onehot = np.concatenate([(onehot_base == j).astype(np.int8) for j in range(n_classes)], axis=-1)
        #  detect_mask           [[GRID_SIZE, GRID_SIZE, N_anchor, 1],
        #  matching_gTruth_boxes  [GRID_SIZE, GRID_SIZE, N_anchor, x-y-w-h-l],
        #  class_onehot           [GRID_SIZE, GRID_SIZE, N_anchor, n_classes],
        #  gTruth_boxes_grid      [N_labels,  x-y-w-h-l]]
        return detect_mask, matching_gTruth_boxes, class_onehot, gTruth_boxes_grid

    @staticmethod
    def image_transform(image):
        # inputs are filenames, read it
        if isinstance(image, str):
            image = imread(image, True, "simpleitk")

        image   =   torch.tensor(image)
        mu      =   torch.mean(image, dim=(1, 2), keepdim=True)
        sigma   =   torch.std(image, dim=(1, 2), keepdim=True)
        image   =   torch.sigmoid((image-mu)/sigma)
        return image

    @staticmethod
    def target_transform(label: np.ndarray, cls_loc=0) -> tuple:
        """
        Parameters
        ---------
        label:
            [N_max, (cls, x, y, w, h)]
        cls_loc: int
            where is the class parameter is
        Returns
        -------
        detect_mask, gt_boxes, class_oh, box_grid
        """
        def toTensor32(inputs: tuple):
            return [torch.from_numpy(x).to(F32) for x in inputs]
        
        return toTensor32(DataTransform.process_truth_boxes(label, ANCHORS, DST_IMGSZ, GRIDSZ, N_CLASSES, cls_loc))
        

class BloodSmearDataset(Dataset):
    """ Loading the given files, produce a Dataset object. """
    def __init__(self, xml_filename="", filepath="", training_stage="train",
                 image_transform=None, target_transform=None, dst_imgsz=DST_IMGSZ):
        self.max_boxes              =   0
        self.image_transform        =   image_transform
        self.target_transform       =   target_transform
        self.destined_image_shape   =   (dst_imgsz, dst_imgsz)

        if xml_filename:    # Load data from an XML file
            self.src_filename = xml_filename
            # Create a container for parsing the raw data loaded from disk,
            # and operating (Splitting) the dataset through attached method.
            spContainer = StandardXMLContainer.fromXML(xml_filename)
            # sample_sets: np.array([MltSample1, MltSample2, MltSample3, ....])
            self.modalities, self.labels = spContainer.dataset_output(self.destined_image_shape)

        elif filepath:
            self.modalities, self.labels = load_modalities_training_data(filepath, training_stage, dst_imgsz)

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        pmd, plbl = list(map(lambda obj: (type(obj[0]), len(obj), obj.shape), (self.modalities, self.labels)))
        return f"<Class 'BloodSmearDataset'>\n\tSource: {self.src_filename}\n\t" \
               f"Modalities: {pmd[0]}*{pmd[1]}, shape-{pmd[2]}\n\tLabels: {plbl[0]}*{plbl[1]}, shape:-{plbl[2]}"

    @classmethod
    def from_xml_cache(cls, filename, image_transform=None, target_transform=None):
        if not os.path.isfile(filename):
            return None

        # Construct instance
        with open(filename, "rb") as f:
            cache_dict = pickle.load(f)
        dataset = cls(image_transform=image_transform, target_transform=target_transform,
                      dst_imgsz=cache_dict["destined_image_shape"])

        # Recall attributes
        dataset.max_boxes   =   cache_dict["max_boxes"]
        dataset.modalities  =   cache_dict["modalities"]
        dataset.labels      =   cache_dict["labels"]
        return dataset

    def __getitem__(self, idx):
        labels      =   self.labels[idx]
        modalities  =   self.modalities[idx]
        # img numpy(320,320) label list(295,5) -> img (320,320) label (295,5)
        # the shape of input images and labels might not corresponding to the image shape of (320, 320)
        if self.image_transform:
            modalities = self.image_transform(modalities)
        if self.target_transform:
            labels = self.target_transform(labels)
        return modalities, labels


def create_dataloader(fpath_str: str, training_stage, batch_size, image_tf=None, target_tf=None, **loader_params):
    constructor = {"image_transform":  DataTransform.image_transform if image_tf is None else image_tf,
                   "target_transform": DataTransform.target_transform if target_tf is None else target_tf}

    if isdir(fpath_str):
        constructor.update({"filepath": fpath_str, "training_stage": training_stage})
    elif isfile(fpath_str):
        if fpath_str.endswith("xml"):
            constructor.update({"xml_filename": fpath_str, "training_stage": training_stage})
        elif fpath_str.endswith("pkl"):
            constructor.update({"filename": fpath_str, "training_stage": training_stage})

    return DataLoader(batch_size=batch_size, dataset=BloodSmearDataset(**constructor), **loader_params)


# Pytorch format Dataset Constructor, using in the initialing of 'BloodSmearDataset'
XML_DATASET_FILENAME    =   join(DATA_ROOT, "20210105 BloodSmear\\fov_annotations.xml")
XML_CACHE_FILENAME      =   ".\\caches\\cache-20210105-blood_smear.pkl"
SET202109_FILENAME      =   join(DATA_ROOT, "Set-202109")


if __name__ == "__main__":
    # Logging Config -----------------------------------------------------------------
    logging.config.fileConfig("..\\log\\config\\dataset.conf")
    logger      =   logging.getLogger(__name__)
    # Logging Config Ended -----------------------------------------------------------
    try:
        bsLoader = create_dataloader(SET202109_FILENAME, "valid", 4, shuffle=True)
        for batch_sample in bsLoader:
            bt_modalities, bt_labels = batch_sample["modalities"], batch_sample["labels"]
            single_sample_visualization(bt_modalities[0].numpy(), bt_labels[3][0].numpy())
    except Exception as err:
        logger.exception(err)
