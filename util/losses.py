import torch
import numpy as np
from collections import ChainMap

from ..util.functions import anchors_compile


def compute_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    # x1...:[b,GRIDSZ,GRIDSZ,5]
    xmin1 = x1 - 0.5 * w1
    xmax1 = x1 + 0.5 * w1
    ymin1 = y1 - 0.5 * h1
    ymax1 = y1 + 0.5 * h1

    xmin2 = x2 - 0.5 * w2
    xmax2 = x2 + 0.5 * w2
    ymin2 = y2 - 0.5 * h2
    ymax2 = y2 + 0.5 * h2

    # (xmin1,ymin1,xmax1,ymax1), (xmin2,ymin2,xmax2,ymax2)
    interw = torch.minimum(xmax1, xmax2) - torch.maximum(xmin1, xmin2)
    interh = torch.minimum(ymax1, ymax2) - torch.maximum(ymin1, ymin2)
    inter = torch.clip(interw, 0) * torch.clip(interh, 0)
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / (union + 1e-6)
    return iou  # [b,GRIDSZ,GRIDSZ,5]


def cartesian_coordinate(batch_size, grid_size):
    """ return cartesian base torch.Tensor """
    x_grid, y_grid = torch.arange(grid_size), torch.arange(grid_size)
    # x_grid, y_grid shape: (GRIDSZ, GRIDSZ)
    y_grid, x_grid = torch.meshgrid(y_grid, x_grid)
    # should expand dim into (1, GRIDSZ, GRIDSZ, 1, 1)  represents: (batch_sz, M, N, n_anchors, xy)
    x_grid = x_grid.view(1, grid_size, grid_size, 1, 1)
    y_grid = y_grid.view(1, grid_size, grid_size, 1, 1)

    xy_grid = torch.cat([x_grid, y_grid], dim=-1)  # (1,GRIDSZ,GRIDSZ,1,2)
    xy_grid = torch.tile(xy_grid, (batch_size, 1, 1, 4, 1))  # (b,GRIDSZ,GRIDSZ,n_anchors,2)
    return xy_grid


class FeatureTranslator:
    """ Comprehend the output of the model in the forward propagation.
    Notice: All the operation here is based on torch module. """
    def __init__(self, outputs: torch.Tensor, anchors, device, grid_size, batch=True):
        self.data = outputs
        self.device = device
        self.gdsz = grid_size
        self.anchors = torch.from_numpy(anchors_compile(anchors)).to(device)
        self.n_anchor = len(self.anchors)
        self.xy, self.wh, self.conf, self.scores = None, None, None, None
        self.interpret(batch)

    def interpret(self, batch):
        # xy: (Batch, Grid_size, Grid_size, N_anchors, x-y)
        x, y = np.meshgrid(np.arange(self.gdsz), np.arange(self.gdsz))
        # (b, Grid_size, Grid_size) + (b, Grid_size, Grid_size) -> (b, Grid_size, Grid_size, 1, 2)
        flag = "4,5,1" if batch else "3,4,0"
        xy_grid = torch.from_numpy(np.r_[flag, x, y]).to(self.device)

        # (b, gdsz, gdsz, N_anchors, x-y) + (1, gdsz, gdsz, 1, 2) -> (b, gdsz, gdsz, N_anchors, x-y)
        self.xy = xy_grid + torch.sigmoid(self.data[..., :2])
        # wh: (Batch, Grid_size, Grid_size, N_anchors, w-h)
        self.wh = self.anchors * torch.exp(self.data[..., 2:4])
        # conf: (Batch, Grid_size, Grid_size, N_anchors, conf)
        self.conf = torch.sigmoid(self.data[..., 4])
        # scores: (Batch, Grid_size, Grid_size, N_anchors, N_classes)
        self.scores = torch.log_softmax(self.data[..., 5:], -1)


class YoloLoss:
    def __init__(self, device, anchors, grid_size, n_classes):
        self.device = device
        self.accuracy = 0
        self.anchors = anchors
        self.grid_size = grid_size
        self.n_classes = n_classes

    @staticmethod
    def masked_mse(a, b, mask, n, eps=1e-6):
        return torch.sum(mask * torch.square(a - b)) / (n + eps)

    @staticmethod
    def coordinate_loss(truth_xy, pred_xy, truth_wh, pred_wh, truth_mask, truth_nobj):
        """ [b,GRIDSZ,GRIDSZ,N_anchors,5+n_classes] x-y-w-h-conf-l0-l1-l2-l3
        Map the predicted bias to the truth value by the grid size
        All of the inputs is torch.Tensor """
        # [b,GRIDSZ,GRIDSZ,N_anchors,2] - [b,GRIDSZ,GRIDSZ,N_anchors,2]
        xy_loss = YoloLoss.masked_mse(truth_xy, pred_xy, truth_mask, truth_nobj)
        # [b,GRIDSZ,GRIDSZ,N_anchors,2] - [b,GRIDSZ,GRIDSZ,N_anchors,2]
        wh_loss = YoloLoss.masked_mse(truth_wh, pred_wh, truth_mask, truth_nobj)
        return xy_loss + wh_loss

    def class_loss(self, truth_classes_oh, truth_mask, truth_nobj, pred_scores):
        # truth_classes_oh: [b, GRID_SIZE, GRID_SIZE, N_anchor, n_classes] => [b, GRID_SIZE, GRID_SIZE, N_anchor]
        true_scores = torch.argmax(truth_classes_oh, -1)
        # the input of CrossEntropyLoss should be (N, C, d1, d2, ...) vs (N, d1, d2, ...)
        pred_scores = pred_scores.permute((0, 4, 1, 2, 3))

        # Define loss compute object
        loss = torch.nn.CrossEntropyLoss(reduction="none", weight=torch.tensor([1., 2., 1.]).to(self.device))
        # Compute loss: [b,n_classes, Grid_size,Grid_size,N_anchors] vs [b,GRID_SIZE,GRID_SIZE,N_anchor]
        class_loss = loss(pred_scores, true_scores)
        # [b,GRIDSZ,GRIDSZ,N_anchors] => [b,GRIDSZ,GRIDSZ,N_anchors,1] * [b,GRIDSZ,GRIDSZ,N_anchors,1]
        class_loss = torch.unsqueeze(class_loss, -1) * truth_mask
        class_loss = torch.sum(class_loss) / (truth_nobj + 1e-6)
        return class_loss

    @staticmethod
    def object_loss(gtruth_boxes, truth_mask, truth_nobj, pred_xy, pred_wh, pred_conf):
        # gtruth_boxes [b, GRID_SIZE, GRID_SIZE, N_anchor, x-y-w-h-l] -> [5, b, GRID_SIZE, GRID_SIZE, N_anchor]
        x1, y1, w1, h1, _ = gtruth_boxes.permute(4, 0, 1, 2, 3)
        # (Batch, Grid_size, Grid_size, N_anchors, x-y)
        x2, y2, w2, h2 = pred_xy[..., 0], pred_xy[..., 1], pred_wh[..., 0], pred_wh[..., 1]
        # [b,GRIDSZ,GRIDSZ,4] -> [b,GRIDSZ,GRIDSZ,4,1]
        ious = compute_iou(x1, y1, w1, h1, x2, y2, w2, h2).unsqueeze(-1)

        accuracy = torch.sum(truth_mask * ious) / (truth_nobj + 1e-6)
        obj_loss = -torch.sum(truth_mask * torch.log(pred_conf)) / (truth_nobj + 1e-6)
        return obj_loss, accuracy

    @staticmethod
    def no_object_loss(truth_boxes_grid, truth_mask, pred_xy, pred_wh, pred_conf, noobj_iou_thres=0.5):
        # Predictions
        # [b,GSZ,GSZ,N_anchor,2] => [b,GSZ,GSZ,N_anchor, 1, 2]
        pred_xy = torch.unsqueeze(pred_xy, dim=4)
        # [b,GSZ,GSZ,N_anchor,2] => [b,GSZ,GSZ,N_anchor, 1, 2]
        pred_wh = torch.unsqueeze(pred_wh, dim=4)
        pred_wh_half = pred_wh / 2.
        pred_xymin = pred_xy - pred_wh_half  # [b,GSZ,GSZ,N_anchor, 1, 2]
        pred_xymax = pred_xy + pred_wh_half  # [b,GSZ,GSZ,N_anchor, 1, 2]

        # Ground Truth
        # [b, n_labels, 5] => [b, 1, 1, 1, n_labels, 5]
        b, n_lbs, len_box = truth_boxes_grid.shape
        true_boxes_grid = truth_boxes_grid.view(b, 1, 1, 1, n_lbs, len_box)
        true_xy = true_boxes_grid[..., 0:2]  # [b, 1, 1, 1, n_labels, 2]
        true_wh = true_boxes_grid[..., 2:4]  # [b, 1, 1, 1, n_labels, 2]
        true_wh_half = true_wh / 2.
        true_xymin = true_xy - true_wh_half
        true_xymax = true_xy + true_wh_half

        # Compute non object loss from predxymin, predxymax, true_xymin, true_xymax
        # [b,GSZ,GSZ,N_anchor,1,2] vs [b,1,1,1,296,2] =>[b,GSZ,GSZ,N_anchor,296,2]
        intersectxymin = torch.maximum(pred_xymin, true_xymin)
        # [b,GSZ,GSZ,N_anchor,1,2] vs [b,1,1,1,296,2] =>[b,GSZ,GSZ,N_anchor,296,2]
        intersectxymax = torch.minimum(pred_xymax, true_xymax)
        # [b,GSZ,GSZ,N_anchor,296,2]
        intersect_wh = torch.maximum(intersectxymax - intersectxymin, torch.zeros_like(intersectxymax))
        # [b,GSZ,GSZ,N_anchor,296]*[b,GSZ,GSZ,N_anchor,296] =>[b,GSZ,GSZ,N_anchor,296]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # [b,GSZ,GSZ,N_anchor] * [b,GSZ,GSZ,N_anchor]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        # [b,1,1,1,296] * [b,1,1,1,296]
        true_area = true_wh[..., 0] * true_wh[..., 1]
        # [b,GSZ,GSZ,N_anchor,1]+[b,1,1,1,296] -[b,GSZ,GSZ,N_anchor,296]=>[b,GSZ,GSZ,N_anchor,296]
        union_area = pred_area + true_area - intersect_area
        # [b,GSZ,GSZ,N_anchor,296]
        iou_score = intersect_area / union_area
        # [b,GSZ,GSZ,N_anchor] => [b,GSZ,GSZ,N_anchor,1]
        best_iou = torch.amax(iou_score, dim=4).unsqueeze(-1)

        noobj_detection = (best_iou < noobj_iou_thres).float()
        noobj_mask = noobj_detection * (1 - truth_mask)

        # noobj counter
        n_noobj = torch.sum((noobj_mask > 0.).to(torch.float32))
        noobj_loss = -torch.sum(noobj_mask * torch.log(1 - pred_conf)) / (n_noobj + 1e-6)
        return noobj_loss

    def __call__(self, y_pred, y_truth, w=None, *args, **kwargs):
        """
        Loss: distance between ground truth and prediction\n
        Parameters
        ----------
        y_pred: torch.Tensor
            [b,GRIDSZ,GRIDSZ,N_anchor,9] x-y-w-h-conf-l0-l1-l2-l3

        y_truth: tuple
            mask: torch.Tensor
                [b,GRIDSZ,GRIDSZ,N_anchor,1]
            gt_boxes: torch.Tensor
                [b,GRIDSZ,GRIDSZ,N_anchor,5] x-y-w-h-l
            classes_oh: torch.Tensor
                [b,GRIDSZ,GRIDSZ,N_anchor,N_classes] l1-l2
            boxes_grid: torch.Tensor
                [b,N_labels,5] x-y-w-h-l
        """
        # Losses Weight
        w = ChainMap(w if w else {}, {"object": 1, "no_obj": .1, "coord": 1, "class": 2})

        # Truth Data
        truth_mask, truth_box_grids, truth_class_onehot, truth_boxes = y_truth
        truth_xy = truth_box_grids[..., :2]     # [b, gsz, gsz, n_anc, 2]
        truth_wh = truth_box_grids[..., 2:4]    # [b, gsz, gsz, n_anc, 2]
        obj_number = torch.sum(truth_mask)      # int

        # Predictions (b, grid_size, grid_size, n_anchors, 5+n_cls)
        pred = FeatureTranslator(y_pred, self.anchors, self.device, self.grid_size)
        pred.conf = pred.conf.unsqueeze(-1)     # (b, gdsz, gdsz, nanc) -> (b, gdsz, gdsz, nanc, 1)

        # Child Losses
        coord_loss = w["coord"] * self.coordinate_loss(truth_xy, pred.xy, truth_wh, pred.wh, truth_mask, obj_number)
        class_loss = w["class"] * self.class_loss(truth_class_onehot, truth_mask, obj_number, pred.scores)
        object_loss, accuracy = self.object_loss(truth_box_grids, truth_mask, obj_number, pred.xy, pred.wh, pred.conf)
        noobj_loss = w["no_obj"] * self.no_object_loss(truth_boxes, truth_mask, pred.xy, pred.wh, pred.conf)
        object_loss *= w["object"]

        self.loss, self.accuracy = coord_loss + class_loss + object_loss + noobj_loss, accuracy
        return self.loss, (coord_loss, object_loss, noobj_loss, class_loss)
