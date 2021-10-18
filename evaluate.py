from os.path import join

import tqdm
import torch
import logging
import numpy as np
from logging        import config
from numpy          import ndarray
from matplotlib     import pyplot as plt

from Deeplearning.util.losses       import Prediction
from Deeplearning.util.dataset      import create_dataloader, ANCHORS, SET202109_FILENAME, GRIDSZ, N_CLASSES
from Deeplearning.util.functions    import locationSize2fourPoints, fourPoints2locationSize, nx

MODEL_PATH = "..\\models"


def softmax(arr):
    arr -= np.max(arr, axis=-1)[..., np.newaxis]
    return np.exp(arr) / np.sum(np.exp(arr), axis=-1)[..., np.newaxis]


def non_maximum_suppress(coord_2d, boxes_conf, class_scores,
                         conf_thresh=.6, over_thresh=.4):
    confidences  =   boxes_conf[boxes_conf > conf_thresh]
    coordinates  =   locationSize2fourPoints(coord_2d)[boxes_conf > conf_thresh]
    class_scores =   softmax(class_scores[boxes_conf > conf_thresh])
    class_scores =   np.argmax(class_scores, axis=-1)

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
        inter   =   w*h
        over    =   inter / (areas[idx] + areas[order[:-1]] - inter)
        order   =   order[:-1][over < over_thresh]

    return fourPoints2locationSize(coordinates[pick]), class_scores[pick]


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
    row_height      =   int(height / (k_split - 2*overrates))
    collum_width    =   int(width / (k_split - 2*overrates))
    overlap_height  =   int(overrates*row_height)
    overlap_width   =   int(overrates*collum_width)

    # position of subview original image on initial image
    split_images = []
    original_point_positions = []
    for k in range(k_split):
        for i in range(k_split):
            # determine original point position of split image
            height_origin = k * (row_height - overlap_height)
            width_origin = i * (collum_width - overlap_width)

            subview_image = image[height_origin:height_origin+row_height, width_origin:width_origin+collum_width]
            split_images.append(subview_image)
            original_point_positions.append((height_origin, width_origin))
    return split_images, original_point_positions


class ModelPrediction(Prediction):
    def __init__(self, pred_tensor, grid_size, anchors, n_class, device, conf_thres):
        super(ModelPrediction, self).__init__(pred_tensor, grid_size, anchors, n_class, device)

        self.out = {}
        self.n_batch = len(self.data)
        self.n_box = len(self.anchors)

        self.nms_along_box()
        self.coord_list, self.classes_pred = non_maximum_suppress(
            torch.cat((self.out["xy"], self.out["wh"]), -1), self.out["conf"].squeeze(),
            self.out["class_scores"], conf_thres)

    def n_class(self, name):
        # TODO: Transplant to PredHandle
        mapping = {"rbc": 0, "wbc": 1, "platelet": 2}
        assert name in mapping.keys()
        return np.sum((self.class_scores == mapping[name]).astype(int))

    def nms_along_box(self):
        # TODO: Transplant to PredHandle
        # (b, gdsz, gdsz, n_anc, 1) -> (b, gdsz, gdsz)
        conf_argmax = self.conf.squeeze(-1).argmax(axis=-1)
        # (b, gdsz, gdsz) -> (b, gdsz, gdsz, n_anc)
        oh_mask = torch.stack([conf_argmax == n for n in range(self.n_box)], -1)

        def boxes_nms(vector, mask):
            return vector[mask].view(self.n_batch, self.grid_size, self.grid_size, -1)

        self.out["xy"] = boxes_nms(self.xy, oh_mask)
        self.out["wh"] = boxes_nms(self.wh, oh_mask)
        self.out["conf"] = boxes_nms(self.conf, oh_mask)
        self.out["class_scores"] = boxes_nms(self.class_scores, oh_mask)


class PredHandle:
    def __init__(self, input_images: torch.Tensor, model, pred_params):
        """ Predicting the red blood cell inside given images."""
        self.model = model
        self.inputs = input_images
        self.pred_tensor = self.model(self.inputs)
        assert len(self.pred_tensor.shape) == 5

        # the dimension of Height start on 1
        self.Ni, self.Nj = self.pred_tensor.shape[1:3]
        self.predictions = ModelPrediction(self.pred_tensor, **pred_params)     # processing

    def __getitem__(self, item) -> ModelPrediction:
        prediction = self.predictions[item]
        if self.inputs.requires_grad:
            self.inputs.detach()
        prediction.inputs = self.inputs.cpu().numpy()[item]
        return prediction


def pred_show(input_images, pred_coords, classes_label, image_fname=""):
    scale = 320 / 40
    for i, img in enumerate(input_images):
        fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ["top", "bottom", "right", "left"]:
            ax.spines[sp].set_visible(False)
        if i == 1:
            for coord, lbl in zip(pred_coords, classes_label):
                x, y, w, h = [elm * scale for elm in coord]
                rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, lw=1.6, fill=False,
                                     color=("green", "blue", "purple")[int(lbl)])
                ax.add_patch(rect)
        img_savename = image_fname + f"_{i}.tif"
        print(img_savename)
        if image_fname:
            plt.savefig(img_savename, dpi=100, bbox_inches="tight", pad_inches=0)
        else:
            plt.show()


def image_recognition(dataloader, model_fpath, pred_params):
    """Testing the trained network models."""
    logger.info("Loading Model...")
    device = pred_params.get("device")
    yolo_model = torch.load(model_fpath)

    # Data processing
    predHandles, yData = [], []
    for modalities, labels in tqdm.tqdm(dataloader):
        X, y_truth = modalities.to(device), [item.to(device) for item in labels]
        predHandles.append(PredHandle(X, yolo_model, pred_params))
        yData.append(y_truth)

    # # Instance Show
    # for nd, pHand in enumerate(predHandles[:3]):
    #     for ni, prediction in enumerate(pHand):
    #         pred_show(vl.sigmoid_hist(prediction.input_images), prediction.coord_list, prediction.cls_label,
    #                   image_fname=join(MODEL_PATH, "plan_8.0\\images\\mod_1014-174441_"+str(nd)+"-"+str(ni)))

    # Statistic
    total_objects_truth, total_rbc_truth, total_wbc_truth, total_platelet_truth = 0, 0, 0, 0
    total_objects_pred, total_rbc_pred, total_wbc_pred, total_platelet_pred = 0, 0, 0, 0
    for pHand, y_truth in zip(predHandles, yData):
        print("")
        y_truth = y_truth[3].to("cpu").numpy()
        obj_truth, rbc_truth, wbc_truth, pla_truth = [], [], [], []
        for sp in y_truth:
            # sp: [n_sample, 5]
            sp_mask = np.sum(sp, -1) > 0
            obj_truth.append(np.sum(sp_mask.astype(int)))
            rbc_truth.append(np.sum(sp[sp_mask][..., 4] == 0.))
            wbc_truth.append(np.sum(sp[sp_mask][..., 4] == 1.))
            pla_truth.append(np.sum(sp[sp_mask][..., 4] == 2.))
        obj_pred = [len(pred.coord_list) for pred in pHand.predictions]  # One batch object number
        rbc_pred = [pred.n_class("rbc") for pred in pHand.predictions]
        wbc_pred = [pred.n_class("wbc") for pred in pHand.predictions]
        pla_pred = [pred.n_class("platelet") for pred in pHand.predictions]
        for num in (obj_truth, rbc_truth, wbc_truth, pla_truth):
            print(num)
        for num in (obj_pred, rbc_pred, wbc_pred, pla_pred):
            print(num)

        total_objects_truth += np.sum(obj_truth)
        total_rbc_truth += np.sum(rbc_truth)
        total_wbc_truth += np.sum(wbc_truth)
        total_platelet_truth += np.sum(pla_truth)

        total_objects_pred += np.sum(obj_pred)
        total_rbc_pred += np.sum(rbc_pred)
        total_wbc_pred += np.sum(wbc_pred)
        total_platelet_pred += np.sum(pla_pred)

    print(f"Truth  total: {total_objects_truth}, rbc {total_rbc_truth}, wbc {total_wbc_truth}, "
          f"platelet {total_platelet_truth}\n"
          f"Predict total: {total_objects_pred}, rbc {total_rbc_pred}, wbc {total_wbc_pred}, "
          f"platelet {total_platelet_pred}\n"
          f"Accuracy total: {100*total_objects_pred/total_objects_truth:.2f} %, "
          f"rbc {100*total_rbc_pred/total_rbc_truth:.2f} %, "
          f"wbc {100*total_wbc_pred/total_wbc_truth:.2f} %, "
          f"platelet {100*total_platelet_pred/total_platelet_truth:.2f} %\n"
          f"Error total: {100*(total_objects_pred/total_objects_truth-1):.2f} %, "
          f"rbc {100*(total_rbc_pred/total_rbc_truth-1):.2f} %, "
          f"wbc {100*(total_wbc_pred/total_wbc_truth-1):.2f} %, "
          f"platelet {100*(total_platelet_pred/total_platelet_truth-1):.2f} %")


if __name__ == '__main__':
    # ================== Python interpreter initializing =================
    if not plt.get_backend() == "tkagg":
        plt.switch_backend("tkagg")
    # Logging Config ---------------------------------------------------
    logging.config.fileConfig(".\\log\\config\\evaluate.conf")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Logging Config Ended ---------------------------------------------

    # ========================== Scripts Execute =========================
    try:
        CONF_THRES      =   0.6
        DEVICE          =   "cuda" if torch.cuda.is_available() else "cpu"
        modelFullname   =   join(MODEL_PATH, "plan_8.0\\yolov5_1014-174441.pth")
        test_loader     =   create_dataloader(SET202109_FILENAME, "test", 8, shuffle=True)
        predParams      =   {"grid_size": GRIDSZ, "anchors": ANCHORS, "n_class": N_CLASSES,
                             "device": DEVICE, "conf_thres": CONF_THRES}

        image_recognition(test_loader, modelFullname, pred_params=predParams)
    except Exception as e:
        logger.exception(e)
