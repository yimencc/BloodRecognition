import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch

import matplotlib.pyplot as plt

from dataset import ANCHORS, GRIDSZ

F32 = torch.float32


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
    inter = interw * interh
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / (union + 1e-6)
    # [b,GRIDSZ,GRIDSZ,5]
    return iou


def _cartesian_coordinate(batch_size):
    """
    return cartesian base torch.Tensor
    """
    x_grid = torch.arange(0, GRIDSZ)
    y_grid = torch.arange(0, GRIDSZ)
    # x_grid, y_grid shape: (GRIDSZ, GRIDSZ)
    y_grid, x_grid = torch.meshgrid(y_grid, x_grid)
    # should expand dim into (1, GRIDSZ, GRIDSZ, 1, 1)
    # represents: (batch_sz, M, N, n_anchors, xy)
    x_grid = x_grid.view(1, GRIDSZ, GRIDSZ, 1, 1)
    y_grid = y_grid.view(1, GRIDSZ, GRIDSZ, 1, 1)
    xy_grid = torch.cat([x_grid, y_grid], dim=-1)  # (1,GRIDSZ,GRIDSZ,1,2)
    xy_grid = torch.tile(xy_grid, (batch_size, 1, 1, 4, 1))
    return xy_grid


class YoloLoss:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.accuracy = 0

    def coordinate_losses(self):
        """
        [b,GRIDSZ,GRIDSZ,4,9] x-y-w-h-conf-l0-l1-l2-l3
        Map the predicted bias to the truth value by the grid size
        All of the inputs is torch.Tensor
        """
        # Prediction in Truth
        # xy predictions [b,GRIDSZ,GRIDSZ,4,2]
        # [b,GRIDSZ,GRIDSZ,4,2] * [4,2] => [b,GRIDSZ,GRIDSZ,4,2]

        # Labels (Ground Truth)
        # how many objects really exist in a batch
        # [b,GRIDSZ,GRIDSZ,4,1] * [b,GRIDSZ,GRIDSZ,4,2]
        xy_loss = torch.square(self.true_xy - self.pred_xy)
        xy_loss = torch.sum(self.mask*xy_loss) / (self.n_obj+1e-6)

        # wh_loss = torch.square(torch.sqrt(self.true_wh) - torch.sqrt(self.pred_wh))
        wh_loss = torch.square(torch.sqrt(self.true_wh)-torch.sqrt(self.pred_wh))
        wh_loss = torch.sum(self.mask*wh_loss) / (self.n_obj+1e-6)

        coord_loss = xy_loss + wh_loss
        return coord_loss

    def classes_loss(self):
        # [b,GRIDSZ,GRIDSZ,4,4]
        # [b,GRIDSZ,GRIDSZ,4]
        true_box_class = torch.argmax(self.classes_oh, -1)

        # Compute loss: [b,GRIDSZ,GRIDSZ,4, 4] vs [b,GRIDSZ,GRIDSZ,4]
        loss = torch.nn.CrossEntropyLoss(reduction="mean")
        class_loss = loss(self.pred_box_class, true_box_class)
        # [b,GRIDSZ,GRIDSZ,4] => [b,GRIDSZ,GRIDSZ,4,1]* [b,GRIDSZ,GRIDSZ,4,1]
        class_loss = torch.unsqueeze(class_loss, -1) * self.mask
        class_loss = torch.sum(class_loss) / (self.n_obj + 1e-6)
        return class_loss

    def object_loss(self):
        # [b,GRIDSZ,GRIDSZ,4,5] -> [b,GRIDSZ,GRIDSZ,4,4]
        x1, y1, w1, h1 = self.gt_boxes.permute(4, 0, 1, 2, 3)[:4]
        # [b,GRIDSZ,GRIDSZ,4,2]*2 -> [b,GRIDSZ,GRIDSZ,4,4]
        x2, y2 = self.pred_xy[..., 0], self.pred_xy[..., 1]
        w2, h2 = self.pred_wh[..., 0], self.pred_wh[..., 1]

        ious = compute_iou(x1, y1, w1, h1, x2, y2, w2, h2).unsqueeze(-1)
        self.accuracy = torch.sum(self.mask*ious)/(self.n_obj+1e-6)
        # [b,GRIDSZ,GRIDSZ,4] -> [b,GRIDSZ,GRIDSZ,4,1]

        obj_loss = torch.sum(self.mask*torch.square(ious-self.pred_conf)
                             )/(self.n_obj + 1e-6)
        return obj_loss

    def non_object_loss(self):
        # Predictions
        # [b,GRIDSZ,GRIDSZ,4,2] => [b,GRIDSZ,GRIDSZ,4, 1, 2]
        pred_xy = torch.unsqueeze(self.pred_xy, dim=4)
        # [b,GRIDSZ,GRIDSZ,4,2] => [b,GRIDSZ,GRIDSZ,4, 1, 2]
        pred_wh = torch.unsqueeze(self.pred_wh, dim=4)
        pred_wh_half = pred_wh / 2.
        pred_xymin = pred_xy - pred_wh_half
        pred_xymax = pred_xy + pred_wh_half

        # Ground Truth
        # [b, 296, 4] => [b, 1, 1, 1, 296, 4]
        b, n, c = self.boxes_grid.shape
        true_boxes_grid = self.boxes_grid.view(b, 1, 1, 1, n, c)
        true_xy = true_boxes_grid[..., 0:2]
        true_wh = true_boxes_grid[..., 2:4]
        true_wh_half = true_wh / 2.
        true_xymin = true_xy - true_wh_half
        true_xymax = true_xy + true_wh_half

        # Compute non object loss
        # predxymin, predxymax, true_xymin, true_xymax
        # [b,GRIDSZ,GRIDSZ,4,1,2] vs [b,1,1,1,296,2]
        # =>[b,GRIDSZ,GRIDSZ,4,296,2]
        intersectxymin = torch.maximum(pred_xymin, true_xymin)
        # [b,GRIDSZ,GRIDSZ,4,1,2] vs [b,1,1,1,296,2]
        # =>[b,GRIDSZ,GRIDSZ,4,296,2]
        intersectxymax = torch.minimum(pred_xymax, true_xymax)
        # [b,GRIDSZ,GRIDSZ,4,296,2]
        intersect_wh = torch.maximum(intersectxymax - intersectxymin,
                                     torch.zeros_like(intersectxymax))
        # [b,GRIDSZ,GRIDSZ,4,296]*[b,GRIDSZ,GRIDSZ,4,296]
        # =>[b,GRIDSZ,GRIDSZ,4,296]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # [b,GRIDSZ,GRIDSZ,4,1]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        # [b,1,1,1,296]
        true_area = true_wh[..., 0] * true_wh[..., 1]
        # [b,GRIDSZ,GRIDSZ,4,1]+[b,1,1,1,296]
        # -[b,GRIDSZ,GRIDSZ,4,296]=>[b,GRIDSZ,GRIDSZ,4,296]
        union_area = pred_area + true_area - intersect_area
        # [b,GRIDSZ,GRIDSZ,4,296]
        iou_score = intersect_area / union_area
        # [b,GRIDSZ,GRIDSZ,4]
        best_iou = torch.amax(iou_score, dim=4).unsqueeze(-1)
        # [b,GRIDSZ,GRIDSZ,4,1]

        nonobj_detection = (best_iou < 0.6).float()
        nonobj_mask = nonobj_detection * (1 - self.mask)
        # nonobj counter
        n_nonobj = torch.sum((nonobj_mask > 0.).to(torch.float32))
        nonobj_loss = (torch.sum(nonobj_mask * torch.square(-self.pred_conf))
                       / (n_nonobj + 1e-6))
        return nonobj_loss

    def __call__(self, y_pred, y_truth, *args, **kwargs):
        """
        Loss: distance between ground truth and prediction\n
        Parameters
        ----------
        y_pred: torch.Tensor
            [b,GRIDSZ,GRIDSZ,4,9] x-y-w-h-conf-l0-l1-l2-l3
        y_truth: tuple
            mask: torch.Tensor
                [b,GRIDSZ,GRIDSZ,4,1]
            gt_boxes: torch.Tensor
                [b,GRIDSZ,GRIDSZ,4,5] x-y-w-h-l
            classes_oh: torch.Tensor
                [b,GRIDSZ,GRIDSZ,4,4] l1-l2
            boxes_grid: torch.Tensor
                [b,296,5] x-y-w-h-l
        """
        # Pre-defining Variables
        self.mask, self.gt_boxes, self.classes_oh, self.boxes_grid = y_truth
        self.n_obj = torch.sum(self.mask)
        self.anchors = torch.tensor(ANCHORS, dtype=F32, device=self.device).view(4, 2)
        self.xy_grid = _cartesian_coordinate(y_pred.shape[0]).to(self.device)

        # Predictions
        self.pred_xy_bias = y_pred[..., :2]
        self.pred_wh_bias = y_pred[..., 2:4]
        self.pred_conf = torch.sigmoid(y_pred[..., 4:5])
        self.pred_xy = torch.sigmoid(self.pred_xy_bias)+self.xy_grid
        self.pred_wh = self.anchors * torch.exp(self.pred_wh_bias)
        self.pred_box_class = y_pred[..., 5:].permute(0, 4, 1, 2, 3)

        # Ground Truth
        self.true_xy = self.gt_boxes[..., :2]
        self.true_wh = self.gt_boxes[..., 2:4]

        self.coord_loss = self.coordinate_losses()
        self.class_loss = self.classes_loss()           # should be smallest, but biggest now
        self.obj_loss = 5*self.object_loss() + self.non_object_loss()
        self.sub_losses = [self.coord_loss, self.class_loss, self.obj_loss]
        self.loss = sum(self.sub_losses)
        return self.loss


def plan_loss_plot(plan_fpath, plt_row, plt_col):
    if not plt.get_backend() == "tkagg":
        plt.switch_backend("tkagg")
    fig, axs = plt.subplots(plt_row, plt_col, constrained_layout=True)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    loss_files = [file for file in os.listdir(plan_fpath)
                  if file.startswith("losses")]
    for i, loss_name in enumerate(loss_files):
        row = i // plt_col
        col = i - row*plt_col
        ax = axs[row, col]
        loss_fullname = os.path.join(plan_fpath, loss_name)

        params = {}
        train_loss = []
        valid_loss = []
        accuracy = []
        with open(loss_fullname, "r") as f:
            container_label = ""
            for line in f:
                if line.startswith(("Train", "Valid", "Acc")):
                    if line.startswith("Train"):
                        container_label = "train"
                    elif line.startswith("Valid"):
                        container_label = "valid"
                    else:
                        container_label = "acc"
                elif line.startswith(("Initial", "Max", "Decay")):
                    key, val = line[:-1].split(":")
                    params.update({key: val})
                else:
                    if container_label == "train":
                        train_loss.append(float(line[:-1]))
                    elif container_label == "valid":
                        valid_loss.append(float(line[:-1]))
                    elif container_label == "acc":
                        accuracy.append(float(line[:-1]))

        title = loss_name.removeprefix("losses_").removesuffix(".txt")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.plot(train_loss, label="Train")
        ax.plot(valid_loss, label="valid")
        ax.scatter(len(valid_loss)-1, valid_loss[-1], c="black")
        ax.annotate("Model loss\n     %.3f" % valid_loss[-1],
                    xy=(len(valid_loss)-1, valid_loss[-1]),
                    xycoords="data", textcoords="offset points",
                    xytext=(-30, 5), fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        for n, key in enumerate(params.keys()):
            ax.annotate("%s: %s" % (key, params[key]),
                        xy=[.2*len(accuracy), .4*train_loss[0]],
                        xycoords="data", textcoords="offset points",
                        xytext=(10, n*10), fontsize=8)
        ax_c = ax.twinx()
        ax_c.tick_params(axis="y", labelcolor="green", labelsize=8)
        ax_c.plot(accuracy, label="Acc", c="green")
        ax_c.scatter(len(accuracy)-1, accuracy[-1], c="red")
        ax_c.annotate("Model Accuracy\n     %.3f" % accuracy[-1],
                      xy=(len(accuracy)-1, accuracy[-1]),
                      xycoords="data", textcoords="offset points",
                      xytext=(-50, -20), fontsize=8)
    plt.show()


if __name__ == '__main__':
    PLAN_PATH = "..\\data\\models\\plan_4.6"
    plan_loss_plot(PLAN_PATH, 2, 3)
