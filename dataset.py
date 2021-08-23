import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import shutil
from os.path import join
from os.path import abspath

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

IMGSZ = 320
GRIDSZ = 40
F32 = torch.float32
IMG_PLUGIN = "simpleitk"
ANCHORS = [1., 1., 1.125, 1.125, 1.25, 1.25, 1.375, 1.375]
PATH = {"root": abspath(".."),
        "data": abspath("..\\data"),
        "code": abspath("..\\deeplearning"),
        "dataset": abspath("..\\data\\dataset"),
        "patches": abspath("..\\data\\2021-01-05"),
        "train_ann": abspath("..\\data\\train\\ann"),
        "train_img": abspath("..\\data\\train\\img"),
        "valid_ann": abspath("..\\data\\valid\\ann"),
        "valid_img": abspath("..\\data\\valid\\img")}

IMG_PATCHES = join(PATH["patches"], "phase_patches")
LBL_PATCHES = join(PATH["patches"], "annotations\\manual_labeled_results")


def dataset_separate(src_path, dst_path):
    ann_dir = join(src_path, "annotations")
    img_dir = join(src_path, "images")
    train_path = join(dst_path, "train")
    valid_path = join(dst_path, "valid")
    for folder in [train_path, valid_path]:
        if not os.path.exists(folder):
            os.mkdir(folder)
            os.mkdir(join(folder, "ann"))
            os.mkdir(join(folder, "img"))
    t = 0
    f = 0
    for i, (ann_name, img_name) in enumerate(zip(os.listdir(ann_dir),
                                                 os.listdir(img_dir))):
        ann_fullname = join(ann_dir, ann_name)
        img_fullname = join(img_dir, img_name)
        p = [0.8, 0.2]
        to_train = np.random.choice([True, False], p=p)
        if to_train:
            # copy img -> dst/train/img_dir
            dst_fullname = join(train_path, "ann\\ann_%04d.txt" % t)
            shutil.copyfile(ann_fullname, dst_fullname)
            # copy ann -> dst/train/ann_dir
            dst_fullname = join(train_path, "img\\img_%04d.png" % t)
            shutil.copyfile(img_fullname, dst_fullname)
            t += 1
        else:
            # copy img -> dst/valid/img_dir
            dst_fullname = join(valid_path, "ann\\ann_%04d.txt" % f)
            shutil.copyfile(ann_fullname, dst_fullname)
            # copy ann -> dst/valid/ann_dir
            dst_fullname = join(valid_path, "img\\img_%04d.png" % f)
            shutil.copyfile(img_fullname, dst_fullname)
            f += 1


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
                    interct = np.minimum(
                        w, anchors[j, 0])*np.minimum(h, anchors[j, 1])
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
                    matching_gt_box[y_coord, x_coord, best_anchor] = \
                        np.array([x, y, w, h, box[4]])

        # [296,5] => [32,32,4,5]
        # [32,32,4,5]
        # [32,32,4,1]
        # [296,5]
        return matching_gt_box, detector_mask, gt_boxes_grid

    @staticmethod
    def img_transform(image):
        image = ToTensor()(image)
        mu = torch.mean(image)
        sigma = torch.std(image)
        image = torch.sigmoid((image-mu)/sigma)
        return image

    @classmethod
    def tgt_transform(cls, label):
        """
        Returns
        -------
        tuple
            mask, gt_box, class_oh, box_grid
        """
        gt_box, mask, grid = map(lambda x: torch.from_numpy(x).to(F32),
                                 cls.process_true_boxes(label, ANCHORS, IMGSZ))
        oh_base = torch.tile(torch.zeros_like(mask), (1, 1, 1, 5))
        class_oh = oh_base.scatter_(-1, gt_box[..., 4:].to(torch.int64), 1)
        return mask, gt_box, class_oh[..., 1:], grid

    @classmethod
    def aug_transform(cls, img: np.ndarray, bboxes: list):
        # image: ndarray (320, 320)
        # bboxes: list (295, 5)
        H, W = img.shape
        fliplr = np.random.choice([True, False], p=[.5, .5])
        flipud = np.random.choice([True, False], p=[.5, .5])
        aug_img = None

        if fliplr:
            aug_img = np.fliplr(img)
            for i, bbox in enumerate(bboxes):
                if any(bbox[:4]):
                    bbox[0] = W-1 - bbox[0]

        if flipud:
            aug_img = np.flipud(aug_img) if aug_img is not None else np.flipud(img)
            for i, bbox in enumerate(bboxes):
                if any(bbox[:4]):
                    bbox[1] = H-1 - bbox[1]
        if aug_img is None:
            aug_img = img
        return aug_img.copy(), bboxes

    @staticmethod
    def aug_plot(image, bboxes):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, cmap="gray")
        for bbox in bboxes:
            x, y, w, h = bbox[:4]
            rect = plt.Rectangle((x-w/2, y-h/2), w, h, color="yellow", fill=False)
            ax.add_patch(rect)
        plt.show()


class RbcDataset(Dataset):
    def __init__(self, ann_dir, img_dir, transform=None,
                 target_transform=None, aug_transform=None):
        self.img_labels = self._load_annotations(ann_dir)
        self.img_dir = img_dir
        self.img_fullnames = [join(self.img_dir, fname) for
                              fname in os.listdir(self.img_dir)]
        self.transform = transform
        self.target_transform = target_transform
        self.aug_transform = aug_transform

    def _load_annotations(self, ann_dir):
        self.max_boxes = 0
        ann_list = []

        def _string_reader(file):
            lines = []
            with open(file, "r") as f:
                for line in f:
                    num_string = line[:-1].split(" ")
                    lines.append([int(elm) for elm in num_string])
            return lines

        for ann in [join(ann_dir, name) for name in os.listdir(ann_dir)]:
            annotations = _string_reader(ann)
            ann_list.append(annotations)
            if len(annotations) > self.max_boxes:
                self.max_boxes = len(annotations)

        boxes = np.zeros((len(ann_list), self.max_boxes, 5))
        for i, annotations in enumerate(ann_list):
            # [N,5]
            img_boxes = np.array(annotations)
            # overwrite the N boxes info
            boxes[i, :img_boxes.shape[0]] = img_boxes
        return boxes

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = imread(self.img_fullnames[idx], True)
        label = self.img_labels[idx]
        # image augmentations
        # img numpy(320,320) label list(295,5) -> img (320,320) label (295,5)
        if self.aug_transform:
            image, label = self.aug_transform(image, label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample


def main():
    rbcDataset = RbcDataset(**TRAIN_DS_CONSTRUCTOR)
    dataloader = DataLoader(rbcDataset, 3, True)
    data_slc = next(iter(dataloader))
    image, label = data_slc["image"], data_slc["label"]
    for lb in label[3][0]:
        if any(lb):
            print(lb[2:4])


TRAIN_DS_CONSTRUCTOR = {
    "ann_dir": PATH["train_ann"],
    "img_dir": PATH["train_img"],
    "transform": DataTransform.img_transform,
    "target_transform": DataTransform.tgt_transform
}

VALID_DS_CONSTRUCTOR = {
    "ann_dir": PATH["valid_ann"],
    "img_dir": PATH["valid_img"],
    "transform": DataTransform.img_transform,
    "target_transform": DataTransform.tgt_transform
}


if __name__ == '__main__':
    main()
