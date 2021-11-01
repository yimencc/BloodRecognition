import torch.nn
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import rank
from skimage.morphology import disk
from skimage.util import img_as_ubyte

from cccode.image import ck
nx = np.newaxis


def image_baseline(image, otsu_disk_radii=200, mean_disk_radii=20):
    # Try rank.otsu filter
    def gaussian_normalize(img): return img.min(), img.max(), img_as_ubyte((img-img.min())/(img.max()-img.min()))
    def anti_gaussian_normalize(img, _min, _max): return img*(_max-_min) + _min
    *extrem, image  =   gaussian_normalize(image)
    local_otsu      =   rank.otsu(image, disk(otsu_disk_radii))
    ub_background   =   rank.mean(image, disk(mean_disk_radii), mask=image <= local_otsu)
    background      =   anti_gaussian_normalize(ub_background/256., *extrem)
    return background


def figure_preproduction(figsize, n_row=1, n_col=1):
    fig, axis = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, constrained_layout=True)
    if n_row*n_col != 1:
        for r in range(n_row):
            for c in range(n_col):
                ax = axis[r, c]
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ["top", "bottom", "left", "right"]:
                    ax.spines[sp].set_visible(False)
    else:
        axis.set_xticks([])
        axis.set_yticks([])
        for sp in ["top", "bottom", "left", "right"]:
            axis.spines[sp].set_visible(False)
    return fig, axis


def softmax(arr):
    arr -= np.max(arr, axis=-1)[..., nx]
    return np.exp(arr) / np.sum(np.exp(arr), axis=-1)[..., nx]


def shape2fourPoints(coord_in: torch.Tensor, image_shape):
    # four point to location size coord_in (n_cred, x-y-w-h) -> coord_out (n_cred, x1-y1-x2-y2)
    x, y, w, h = coord_in.permute(1, 0)  # (n_cred,)
    x_max, y_max = image_shape
    x1 = torch.clamp(x-w/2, min=0)
    x2 = torch.clamp(x+w/2, max=x_max-1)
    y1 = torch.clamp(y-h/2, min=0)
    y2 = torch.clamp(y+h/2, max=y_max-1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def fourPoints2shape(coords):
    # PointPoint to PointSize, coords: (n_cred, x1-y1-x2-y2)
    x1, y1, x2, y2 = coords.permute(1, 0)   # (n_cred,)
    x, y = (x1+x2)/2, (y1+y2)/2
    w, h = x2-x1, y2-y1
    return torch.stack([x, y, w, h], dim=-1)


def image_splitting(image, k_split=3, overrates=0.2):
    """ Splitting the image to several subview images with overlapped area,
     image numbers is decided by param k_split, which indicate how mach row
     and collum will the image be divided to.
     Returns
        split images: list
            (k_split * k_split)
        original_point_position: list
            (k_split * k_split), the positions of
            split image original point (usually TopLeft) on the initial image.
     """
    if len(image.shape) == 3:
        height, width = image.shape[:2]
    else:   # image only have two dimension
        height, width = image.shape

    # size of subview image (split image)
    col_width = int(width / (k_split - 2 * overrates))
    row_height = int(height / (k_split - 2 * overrates))
    overlap_width = int(overrates * col_width)
    overlap_height = int(overrates * row_height)

    # position of subview original image on initial image
    split_images = []
    original_point_positions = []
    for k in range(k_split):
        for i in range(k_split):
            # determine original point position of split image
            height_origin = k * (row_height - overlap_height)
            width_origin = i * (col_width - overlap_width)

            subview_image = image[height_origin:height_origin+row_height, width_origin:width_origin+col_width]
            split_images.append(subview_image)
            original_point_positions.append((height_origin, width_origin))
    return split_images, original_point_positions


def histograph(inputs, bins=256):
    hist, edges = np.histogram(inputs, bins=bins)
    centers = [(ef+eb)/2 for ef, eb in zip(edges[:-1], edges[1:])]
    plt.subplots(figsize=(6, 6), constrained_layout=True)
    plt.plot(centers, hist)
    plt.show()


def bboxes_visulization(bboxes, image_shape):
    background = np.ones(image_shape)
    background[1, 1] = 0
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    plt.imshow(background, cmap="gray")
    for x, y, w, h in bboxes:
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, fill=False, color="green")
        ax.add_patch(rect)
    plt.show()


def single_sample_visualization(modality, labels, scale=8):
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
            for x, y, w, h, cls in labels:
                # (x, y, w, h) is the coordinates on the y label plane, sizeof (40, 40)
                if scale:
                    x, y, w, h = [elm * scale for elm in (x, y, w, h)]
                color = ("green", "blue", "yellow", "red", "pink")[int(cls)]
                rect = plt.Rectangle((x - w // 2, y - h // 2), w, h, color=color, fill=False)
                ax.add_patch(rect)
    plt.show()


def model_inspect(model: torch.nn.Module):
    # to cpu numpy
    md_parameters = [(param.to("cpu") if param.device.type == "cuda" else param).detach().numpy().reshape(-1)
                     for param in model.parameters()]

    # inspect
    for i, param in enumerate(md_parameters):
        assert isinstance(param, np.ndarray)
        print(f"{i}, mean: {np.mean(param)}, std: {np.std(param)}")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    fig.suptitle("Model parameters")
    ax.violinplot(md_parameters, showmeans=True, showmedians=False, showextrema=True)
    plt.show()


def model_grads_inspect(model: torch.nn.Module):
    gradients = []
    for param in model.parameters():
        gradients.append(param.grad.to("cpu").detach().numpy().reshape(-1))

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    fig.suptitle("Model parameters")
    ax.violinplot(gradients, showmeans=True, showmedians=False, showextrema=True)
    plt.show()


def yolov5_prediction_inspect(predict: torch.Tensor):
    # predict shape: torch.Size([8, 40, 40, 4, 8)
    # Notice: isinstance(torch.Size, tuple) -> True
    assert len(predict.shape) == 5      # (b, gs, gs, n_anc, 4+1+n_cls)
    n_anchors, len_box = predict.shape[-2:]

    if predict.device.type == "cuda":
        predict = predict.to("cpu")

    inspected_data = predict.detach().numpy().reshape(-1, len_box)

    # Decomposition
    xy          =   inspected_data[:, 2].reshape(-1)
    wh          =   inspected_data[:, :2].reshape(-1)
    confidence  =   inspected_data[:, 4]
    cls_preds   =   inspected_data[:, 5:].reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    fig.suptitle("Predictions xy-wh-conf-cls")
    ax.violinplot((xy, wh, confidence, cls_preds), showmeans=True, showmedians=False, showextrema=True)
    plt.show()
