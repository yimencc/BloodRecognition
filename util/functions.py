from os import listdir
from typing import Union
from os.path import splitext, join

import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from cv2 import resize
from matplotlib.axes import Axes

from scipy.io import loadmat
from skimage.io import imread
from skimage.filters import rank
from skimage.morphology import disk
from skimage.util import img_as_ubyte

nx = np.newaxis
DATA_ROOT = "D:\\Workspace\\Blood Recognition\\datasets"


def iou_assembling(coordinates):
    na = 0
    assemble_indexes = []
    residual_indexes = list(range(len(coordinates)))
    while len(residual_indexes) > 0:  # Still some coordinate didn't decided
        k = 0  # Iterating times for this assemble
        assemble_indexes.append([residual_indexes[0]])
        residual_indexes.pop(0)
        while len(assemble_indexes[na]) > k:
            for idx in residual_indexes:
                tgt_coord = coordinates[idx]
                src_coord = coordinates[assemble_indexes[na][k]]
                # Compute IoUs
                xmin1, xmax1 = tgt_coord[0]-tgt_coord[2]/2, tgt_coord[0]+tgt_coord[2]/2
                ymin1, ymax1 = tgt_coord[1]-tgt_coord[3]/2, tgt_coord[1]+tgt_coord[3]/2
                xmin2, xmax2 = src_coord[0]-src_coord[2]/2, src_coord[0]+src_coord[2]/2
                ymin2, ymax2 = src_coord[1]-src_coord[3]/2, src_coord[1]+src_coord[3]/2
                interw = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
                interh = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
                inter = np.clip(interw, 0, None) * np.clip(interh, 0, None)
                if inter > 0:
                    print(f"platelet[{idx}] in assamble[{na}], "
                          f"ecoord:[{xmin1:.1f}, {xmax1:.1f}, {ymin1:.1f}, {ymax1:.1f}], "
                          f"pcoord:[{xmin2:.1f}, {xmax2:.1f}, {ymin2:.1f}, {ymax2:.1f}], "
                          f"w:{interw:.1f},h:{interh:.1f},s:{inter:.1f}")
                    assemble_indexes[na].append(idx)
                    residual_indexes.remove(idx)
            k += 1
        # print(f"Assemble[{na}]:", assemble_indexes[na])
        na += 1
    # print("\nTotal assembles: ", assemble_indexes)
    assembles = []
    for index in assemble_indexes:
        assembles.append(coordinates[index])
    return assembles


def boxes_area_merge(boxes: np.ndarray):
    # boxes: (nboxes, x-y-w-h-l)
    x, y, w, h, lbl = np.transpose(boxes)
    assert len({*lbl}) == 1
    x_min, x_max = np.min(x-w/2), np.max(x+w/2)
    y_min, y_max = np.min(y-h/2), np.max(y+h/2)
    x1, y1 = (x_min+x_max)/2, (y_min+y_max)/2
    w1, h1 = (x_max-x_min), (y_max-y_min)
    return x1, y1, w1, h1, lbl[0]


def anchors_compile(anchors: Union[list, tuple]) -> np.ndarray:
    anchors = np.array(anchors)
    if len(anchors.shape) == 1:
        anchors = anchors.reshape((-1, 2))
    return anchors


def load_txt_list(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(int(line.rstrip("\n")))
    return data


def image_baseline(image, otsu_disk_radii=200, mean_disk_radii=20, scale=1):
    # Try rank.otsu filter
    def gaussian_normalize(img):
        return img.min(), img.max(), img_as_ubyte((img - img.min()) / (img.max() - img.min()))

    def anti_gaussian_normalize(img, _min, _max):
        return img * (_max - _min) + _min

    def estimate_background(_img):
        *extrem, _img = gaussian_normalize(_img)
        local_otsu = rank.otsu(_img, disk(otsu_disk_radii))
        ub_background = rank.mean(_img, disk(mean_disk_radii), mask=_img <= local_otsu)
        return anti_gaussian_normalize(ub_background / 256., *extrem)

    if scale < 1:
        imshape = image.T.shape
        image = resize(image, tuple([int(sz*scale) for sz in image.T.shape]))
        background = estimate_background(image)
        background = resize(background, imshape)
    else:
        background = estimate_background(image)
    return background


def figure_preproduction(n_row=1, n_col=1, figsize=None, constrained_layout=True,
                         spines=False, ticks=False, suptitle=None):
    fig, axis = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, constrained_layout=constrained_layout)
    if suptitle:
        plt.suptitle(suptitle)
    if n_row * n_col != 1:
        for r in range(n_row):
            for c in range(n_col):
                if n_row != 1:
                    ax = axis[r, c]
                else:
                    ax = axis[c]
                if not ticks:
                    ax.set_xticks([])
                    ax.set_yticks([])
                if not spines:
                    for sp in ["top", "bottom", "left", "right"]:
                        ax.spines[sp].set_visible(False)
    else:
        if not ticks:
            axis.set_xticks([])
            axis.set_yticks([])
        if not spines:
            for sp in ["top", "bottom", "left", "right"]:
                axis.spines[sp].set_visible(False)
    return fig, axis


def axis_rectangle_labeling(axes: Axes, coordinates, colors, lw=None, fill=False):
    """ Labeling axes with given bounding boxes. """
    for (x, y, w, h), c in zip(coordinates, colors):
        rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, color=c, fill=fill, linewidth=lw)
        axes.add_patch(rect)
    return axes


def parse_sample_id(sample_id):
    sample_id = splitext(sample_id)[0]
    dataset_date, sample_id = sample_id.split("-")
    subset_id = sample_id[0]
    image_id = sample_id[1:]
    return dataset_date, int(subset_id), int(image_id)


def source_from_sample_id(sample_id, returns="phase", dst_size=(320, 320), root=DATA_ROOT):
    """ Parsing the filename and return the original position fpath of corresponding sample.
     e.g. 20210902-10000.bmp
     """
    dataset_name_pattern = "%s BloodSmear%02d"
    dataset_date, subset_id, image_id = parse_sample_id(sample_id)
    dst_fullpath = join(root, dataset_name_pattern % (dataset_date, subset_id), returns)
    imgid, pthid = divmod(image_id, 9)
    dst_filename = join(dst_fullpath, listdir(dst_fullpath)[imgid])

    if dst_filename.endswith(("bmp", "jpg")):
        data = imread(dst_filename, True, "simpleitk")
    elif dst_filename.endswith("mat"):
        data = loadmat(dst_filename)["phase"]
    else:
        raise Exception

    h_image, w_image = data.shape
    h_chunk, w_chunk = (340, 340)
    centroid_x_arrange = w_image - w_chunk
    centroid_y_arrange = h_image - h_chunk
    centroid_x_interval = centroid_x_arrange // (3 - 1)
    centroid_y_interval = centroid_y_arrange // (3 - 1)

    image_chunks = []
    for i in range(3):  # ROW split
        for k in range(3):  # COLUMN split
            x_centroid = w_chunk // 2 + k * centroid_x_interval
            y_centroid = h_chunk // 2 + i * centroid_y_interval
            x0, y0 = x_centroid - w_chunk // 2, y_centroid - h_chunk // 2
            x1, y1 = x_centroid + w_chunk // 2, y_centroid + h_chunk // 2
            image_chunks.append(data[y0:y1, x0:x1])
    return resize(image_chunks[pthid], dst_size)


def shape2fourpoints(coord_in: torch.Tensor, image_shape):
    # four point to location size coord_in (n_cred, x-y-w-h) -> coord_out (n_cred, x1-y1-x2-y2)
    x, y, w, h = coord_in.permute(1, 0)  # (n_cred,)
    x_max, y_max = image_shape
    x1 = torch.clamp(x - w / 2, min=0)
    x2 = torch.clamp(x + w / 2, max=x_max - 1)
    y1 = torch.clamp(y - h / 2, min=0)
    y2 = torch.clamp(y + h / 2, max=y_max - 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def fourpoints2shape(coords):
    # PointPoint to PointSize, coords: (n_cred, x1-y1-x2-y2)
    x1, y1, x2, y2 = coords.permute(1, 0)  # (n_cred,)
    x, y = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    return torch.stack([x, y, w, h], dim=-1)


def bboxes_visulization(bboxes, image_shape):
    background = np.ones(image_shape)
    background[1, 1] = 0
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    plt.imshow(background, cmap="gray")
    for x, y, w, h in bboxes:
        rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=False, color="green")
        ax.add_patch(rect)
    plt.show()


def single_sample_visualization(modality, boxes, scale=8):
    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(6, 6))
    for i, arr in enumerate(modality):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        ax.imshow(arr, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ["top", "bottom", "left", "right"]:
            ax.spines[sp].set_visible(False)
        if i == 1:
            for x, y, w, h, cls in boxes:
                # (x, y, w, h) is the coordinates on the y label plane, sizeof (40, 40)
                if scale:
                    x, y, w, h = [elm * scale for elm in (x, y, w, h)]
                color = ("green", "blue", "yellow", "red", "pink")[int(cls)]
                rect = plt.Rectangle((x - w // 2, y - h // 2), w, h, color=color, fill=False)
                ax.add_patch(rect)
    plt.show()
