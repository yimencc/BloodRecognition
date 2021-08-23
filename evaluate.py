from os.path import join

import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from skimage.io import imread
from skimage.transform import resize

from dataset import ANCHORS
from models import YoloV4Model
from cccode.math import sigmoid
from cccode.image import Value, Check

ck = Check()
vl = Value()
nx = np.newaxis
MODEL_PATH = "..\\data\\models"


def locationSize2fourPoints(coord_in):
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


def non_maximum_suppress(coord_2d: ndarray, boxes_conf: ndarray, conf_thresh=.6, over_thresh=.4) -> ndarray:
    confidences: np.ndarray = boxes_conf[boxes_conf > conf_thresh]
    coordinates: np.ndarray = locationSize2fourPoints(coord_2d)[boxes_conf > conf_thresh]

    x1, y1, x2, y2 = np.moveaxis(coordinates, -1, 0)
    areas = (x2-x1) * (y2-y1)
    order = confidences.argsort()

    pick = []
    counter = 0
    while order.size > 0:
        idx = order[-1]
        pick.append(idx)
        counter += 1

        xx1 = np.maximum(x1[idx], x1[order[:-1]])
        yy1 = np.maximum(y1[idx], y1[order[:-1]])
        xx2 = np.minimum(x2[idx], x2[order[:-1]])
        yy2 = np.minimum(y2[idx], y2[order[:-1]])

        w = np.maximum(0., xx2-xx1)
        h = np.maximum(0., yy2-yy1)
        inter = w*h
        over = inter / (areas[idx] + areas[order[:-1]] - inter)
        order = order[:-1][over < over_thresh]

    return fourPoints2locationSize(coordinates[pick])


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
        height = image.shape[0]
        width = image.shape[1]
    else:   # image only have two dimension
        height, width = image.shape

    # size of subview image (split image)
    row_height = int(height / (k_split - 2*overrates))
    collum_width = int(width / (k_split - 2*overrates))
    overlap_height = int(overrates*row_height)
    overlap_width = int(overrates*collum_width)

    # position of subview original image on initial image
    split_images = []
    original_point_positions = []
    for k in range(k_split):
        for i in range(k_split):
            # determine original point position of split image
            height_origin = k * (row_height - overlap_height)
            width_origin = i * (collum_width - overlap_width)

            subview_image = image[height_origin:height_origin+row_height,
                                  width_origin:width_origin+collum_width]
            split_images.append(subview_image)
            original_point_positions.append((height_origin, width_origin))
    return split_images, original_point_positions


class PredContainer:
    def __init__(self, pred_tensor, anchors, conf_thresh=.6):
        self.conf_map = pred_tensor[..., 4:5]
        self.cls = pred_tensor[..., 5:]
        self.xy = pred_tensor[..., :2]
        self.wh = pred_tensor[..., 2:4]
        self.anchors = anchors
        self.no_bias()
        self.nms_along_box()

        self.coord_list = None
        self.input_image = None
        self.coord_list = non_maximum_suppress(self.get_coords(), self.conf_map.squeeze(), conf_thresh=conf_thresh)

    @staticmethod
    def _truth_xy(xy: np.ndarray):
        # self.preds (40, 40, 4, 9)
        x_g, y_g = np.meshgrid(np.arange(xy.shape[0]), np.arange(xy.shape[1]))
        # (40, 40) + (40, 40) -> (40, 40, 1, 2)
        xy_grid = np.r_["3,4,0", x_g, y_g]
        # (40, 40, 1, 2) + (40, 40, 4, 2) -> (40, 40, 4, 2)
        return sigmoid(xy) + xy_grid

    def _truth_wh(self, wh):
        if not isinstance(self.anchors, np.ndarray):
            self.anchors = np.array(self.anchors)
        if self.anchors.shape != (4, 2):
            self.anchors = self.anchors.reshape((4, 2))
        # (4, 2) * (b, 40, 40, 4, 2) -> (b, 40, 40, 4, 2)
        return self.anchors*np.exp(wh)

    def get_coords(self):
        return np.r_["-1", self.xy, self.wh]

    def no_bias(self):
        # (40, 40, 4, 2)
        self.xy = self._truth_xy(self.xy)
        # (40, 40, 4, 2)
        self.wh = self._truth_wh(self.wh)
        # (40, 40, 4, 1)
        self.conf_map = sigmoid(self.conf_map)

    def nms_along_box(self):
        # (40, 40, 4, 1) -> (40, 40)
        conf_argmax = self.conf_map.squeeze(-1).argmax(axis=2)
        # (40, 40) -> (40, 40, 4)
        oh_mask = np.concatenate([(conf_argmax == n)[..., nx] for n in range(4)], -1)

        def boxes_nms(vector: np.ndarray, mask: np.ndarray):
            shape_list = list(vector.shape[:2]) + [vector.shape[-1]]
            return vector[mask].reshape(shape_list)

        for elm in ["xy", "wh", "conf_map", "cls"]:
            self.__dict__[elm] = boxes_nms(self.__dict__[elm], oh_mask)


class PredHandle:
    def __init__(self, input_images: torch.Tensor, model, anchors: ndarray, dim_h=1):
        """ Predicting the red blood cell inside given images."""
        self.model = model
        self.anchors = anchors
        self.input_images = input_images

        self.preds = self.model_predict()

        # the dimension of Height start on 1
        self.dim_H = dim_h
        self.Ni, self.Nj = self.preds.shape[1:3]

        # processing
        self.pContainers = self.prediction_splitting()

    def model_predict(self) -> np.ndarray:
        predictions = self.model(self.input_images)
        if predictions.requires_grad:
            predictions = predictions.detach()
        return predictions.numpy()

    def get_input_image(self, index):
        if self.input_images.requires_grad:
            self.input_images.detach()
        return self.input_images.numpy()[index, 0]

    def prediction_splitting(self) -> tuple[PredContainer]:
        """ Separate bundled predictions """
        return tuple([PredContainer(pr, self.anchors) for pr in self.preds])

    def __getitem__(self, item) -> PredContainer:
        pContainer = self.pContainers[item]
        pContainer.input_image = self.get_input_image(item)
        return pContainer


def specific_image_recognition():
    """Testing the trained network models."""

    def load_model(plan_name="plan_5.3", model_name="yolov2_0506-195652.pth"):
        model = YoloV4Model()
        cur_model_fname = join(MODEL_PATH, plan_name, model_name)
        model.load_state_dict(torch.load(cur_model_fname))
        return model

    def load_data(image_fullname):
        image = imread(image_fullname)
        split_images, original_positions = image_splitting(image)
        return split_images, original_positions

    def pred_show(input_image, pred_coords, image_fname=""):
        scale = 320 / 40
        fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
        ax.imshow(input_image, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ["top", "bottom", "right", "left"]:
            ax.spines[sp].set_visible(False)
        for coord in pred_coords:
            x, y, w, h = [elm * scale for elm in coord]
            rect = plt.Circle((x, y), radius=np.maximum(w, h)/2, lw=1.6, fill=False, color="red", alpha=.8)
            ax.add_patch(rect)
        if image_fname:
            plt.savefig(image_fname)
        else:
            plt.show()

    yolo_model = load_model()
    imageFullname = "D:\\Workspace\\RBC Recognition\\data\\2021-01-05\\phase\\pha_70.tif"

    for i, (phase_img, _) in enumerate(zip(*load_data(imageFullname))):
        # the size of split image is (356, 356), which is not satisfy
        # the shape of (320, 320) for network input, therefore should be resized
        # Notice: the size of RBC in resized image will be decreased, other words, not the real size.
        phase_img = resize(phase_img, (320, 320))
        input_tensor = torch.from_numpy(phase_img[nx, nx, ...].astype(np.float32)/255)
        pHand = PredHandle(input_tensor, yolo_model, ANCHORS)
        pred_show(vl.sigmoid_hist(pHand[0].input_image), pHand[0].coord_list)


if __name__ == '__main__':
    if not plt.get_backend() == "tkagg":
        plt.switch_backend("tkagg")
    specific_image_recognition()
