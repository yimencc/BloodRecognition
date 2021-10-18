import torch.nn
import numpy as np
import matplotlib.pyplot as plt

from cccode.image   import Value, Check

ck = Check()
vl = Value()
nx = np.newaxis


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


def locationSize2fourPoints(coord_in: np.ndarray):
    # four point to location size
    # coord_in (40, 40, x-y-w-h) -> coord_out (40, 40, x1-y1-x2-y2)
    x = coord_in[..., 0]    # (40, 40)
    y = coord_in[..., 1]    # (40, 40)
    w = coord_in[..., 2]    # (40, 40)
    h = coord_in[..., 3]    # (40, 40)

    x1 = np.maximum(x-w/2, 0.)
    x2 = np.minimum(x+w/2, 40.)
    y1 = np.maximum(y-h/2, 0.)
    y2 = np.minimum(y+h/2, 40.)
    return np.r_["2,3,0", x1, y1, x2, y2]


def fourPoints2locationSize(coords):
    # PointPoint to PointSize
    # coords: (N, x1-y1-x2-y2)
    x1 = coords[..., 0]    # (40, 40)
    y1 = coords[..., 1]    # (40, 40)
    x2 = coords[..., 2]    # (40, 40)
    y2 = coords[..., 3]    # (40, 40)

    x = (x1+x2)/2
    y = (y1+y2)/2
    w = x2-x1
    h = y2-y1
    return np.r_["1,2,0", x, y, w, h]
