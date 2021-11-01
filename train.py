import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import logging
from logging import config
from os.path import join, abspath

import yaml
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Deeplearning.util.losses   import YoloLoss
from Deeplearning.util.models   import YoloV6Model
from Deeplearning.util.dataset  import create_dataloader

# Meta-Parameters
CLASS_LOC   =   0
N_CLASSES   =   3
DST_IMGSZ   =   320
SEED        =   123456
IMG_PLUGIN  =   "simpleitk"
MODEL_PATH  =   "..\\models"
DEVICE      =   "cuda" if torch.cuda.is_available() else "cpu"

# Fixed the random states
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class _LossTracer:
    def __init__(self, name: str):
        self.name = name
        self.box = []
        self.counter = 0

    def track(self, value, c=None):
        self.box.append(value)
        if c is not None:
            self.counter += c

    def sum(self):
        return sum(self.box)

    def mean(self):
        if self.counter != 0:
            return np.mean(self.box)
        else:
            return self.sum()/self.counter

    def update_state(self):
        self.box = []

    def __getitem__(self, item):
        return self.box[item]


class Tracer:
    def __init__(self, name):
        self.name = name
        self.boxes = {}

    def register_box(self, name=""):
        self.boxes[name] = _LossTracer(name)

    def track(self, value, name, c=None):
        self.boxes[name].track(value, c)

    def update_state(self, name):
        self.boxes[name].update_state()

    def get(self, name, idx):
        return self.boxes[name][idx]


class EarlyStopping:
    """ Early stopping to stop the training when the loss does not improve after
    certain epochs. """
    def __init__(self, patience=5, min_delta=0, mode: str = "max"):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
        self.mode = mode

    def step(self, val_acc):
        if self.best_acc is None:
            self.best_acc = val_acc
        else:
            equation = self.best_acc-val_acc-self.min_delta
            if self.mode == "min":
                expression = equation > 0
            elif self.mode == "max":
                expression = equation < 0
            else:
                raise Exception("Wrong mode")

            if expression:
                self.best_acc = val_acc
                self.counter = 0
            else:
                self.counter += 1
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
                if self.counter >= self.patience:
                    print('INFO: Early stopped')
                    self.early_stop = True


def train_loop(dataloader, model, loss_fn, optimizer, device, decay_rate=0):
    size = len(dataloader.dataset)
    train_loss, model_loss, regula_loss, n_current = 0, 0, 0, 0
    for batch, (modalities, labels) in enumerate(dataloader):
        # Transfer data to gpu
        X, y = modalities.to(device), [item.to(device) for item in labels]

        # Forward propagation
        pred = model(X)

        # Compute training loss
        reg_loss = decay_rate * sum([torch.sum(param.data) for param in model.parameters()])
        total_loss = loss_fn(pred, y) + reg_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Batch loss
        n_current   +=  len(X)
        cur_loss    =   total_loss.item()
        reg_loss    =   reg_loss.item()

        # Epoch loss update
        train_loss  +=  cur_loss*len(X)
        model_loss  +=  (cur_loss-reg_loss)*len(X)
        regula_loss +=  reg_loss*len(X)

        if (batch+1) % 10 == 0:
            print(f"Total: {cur_loss:>7f}   Model: {cur_loss-reg_loss:>7f}   "
                  f"Regula: {reg_loss:>7f}   [{n_current:>5d}/{size:>5d}]")

    avg_train_loss = train_loss/size
    avg_model_loss = model_loss/size
    avg_regula_loss = regula_loss/size
    print(f"Train Error: [Avg Loss {avg_train_loss:>8f}, Avg Model {avg_model_loss:>8f}, "
          f"Avg Regula {avg_regula_loss:>8f}]")
    return avg_train_loss, avg_model_loss, avg_regula_loss


def valid_loop(dataloader, model, loss_fn, device):
    print("\nTest\n"+"-"*30)
    test_loss, accuracy, n_current, size = 0, 0, 0, len(dataloader.dataset)
    with torch.no_grad():
        for batch, (modalities, labels) in enumerate(dataloader):
            # Transfer data to gpu
            X, y = modalities.to(device), [item.to(device) for item in labels]

            # Forward propagation
            pred = model(X)

            # Compute batch test loss
            cur_loos    =   loss_fn(pred, y).item()
            cur_acy     =   loss_fn.accuracy.item()

            # Epoch data
            n_current   +=  len(X)
            test_loss   +=  cur_loos * len(X)
            accuracy    +=  cur_acy * len(X)
            if (batch+1) % 5 == 0:
                print(f"loss: {cur_loos:>7f}  accuracy: {cur_acy:>5f}  [{n_current:>5d}/{size:>5d}]")

    test_loss /= size
    accuracy /= size
    print(f"Test Error: [Avg loss {test_loss:>8f}, Avg accuracy {accuracy:>3f}]\n")
    return test_loss, accuracy


class Trainer:
    def __init__(self, hyp, train_loader=None, valid_loader=None):
        if isinstance(hyp, str) and hyp.endswith(".yaml"):
            with open(hyp, "r") as hyp_file:
                hyp = yaml.load(hyp_file, yaml.FullLoader)

        params_names        =   ("n_box", "n_cls", "grid_size", "anchors", "attention_layer", "lr")
        n_box, n_cls, grid_sz, anchors, attention, lr = [hyp.get(item) for item in params_names]
        self.hyp            =   hyp
        self.stop_epoch     =   0
        self.callbacks      =   {}
        self.train_break    =   False
        self.device         =   hyp.get("device", DEVICE)

        self.loss_fn        =   YoloLoss(self.device, anchors, grid_sz, n_cls)
        self.model          =   YoloV6Model(n_box, n_cls, grid_sz).to(self.device)
        self.optimizer      =   torch.optim.Adam(self.model.parameters(), lr=lr)

        self.train_tracer   =   Tracer("train")
        self.valid_tracer   =   _LossTracer("valid")
        self.acc_tracer     =   _LossTracer("acc")
        self.train_tracer.register_box("total")
        self.train_tracer.register_box("model")
        self.train_tracer.register_box("regula")

        # Register Default Callback
        dft_callback = {"ReduceLR": ReduceLROnPlateau, "EarlyStopping": EarlyStopping}
        self.register_callbacks(schedule=(dft_callback["ReduceLR"], {"optimizer": self.optimizer, "mode": "min"}),
                                early_stopping=(dft_callback["EarlyStopping"], {"patience": 10}))
        # Load data
        logger.info("Loading data ...")
        self.train_loader = create_dataloader(**hyp.get("train_loader_params")) if not train_loader else train_loader
        self.valid_loader = create_dataloader(**hyp.get("valid_loader_params")) if not valid_loader else valid_loader

    def register_callbacks(self, **callbacks):
        """ callbacks: {"cb_name": (callback, params_dict)} """
        for cb_name, (callback, param_dict) in callbacks.items():
            assert isinstance(cb_name, str)
            assert isinstance(param_dict, dict)
            self.callbacks.update({cb_name: callback(**param_dict)})

    def on_train_begin(self, **kwargs):
        # Hyper-params
        print("\nHyper-parameters:\n"+"="*30)
        for k, v in self.hyp.items():
            print(f"\t{k}: {v}")

        # Model Info
        if kwargs.get("model_info", None):
            print("\nModel Info")
            from torchsummary import summary
            summary(self.model, (4, 320, 320))
            print()

        logger.info("Training Start\n")

    def on_train_end(self, **kwargs):
        # Model Save
        if model_fpath := self.hyp.get("model_fpath", None):
            model_prefix = kwargs.get("model_prefix", "yolov5")
            self.model_save(model_fpath, model_prefix)

    @staticmethod
    def on_epoch_begin(**kwargs):
        # Epoch Message
        if epoch := kwargs.get("epoch", None):
            print(f"[Epoch {epoch}]\n" + "-" * 30)

    def on_epoch_end(self, **kwargs):
        """ Mainly for call back executing """
        self.stop_epoch += 1

        # Learning Rate Schedule
        if schedule := self.callbacks.get("schedule", None):
            val_loss = kwargs.get("valid_loss")
            schedule.step(val_loss)

        # Early Stopping
        if early_stopping := self.callbacks.get("early_stopping"):
            accuracy = kwargs.get("accuracy")
            early_stopping.step(accuracy)
            if early_stopping.early_stop:
                logger.info(f"Early stopped at Accuracy {accuracy:.4f}")
                return True

    def training(self, epochs):
        self.on_train_begin(model_info=True)
        for t in range(epochs):
            self.on_epoch_begin(epoch=t + 1)
            # Train and Valid
            tt_loss, md_loss, reg_loss = train_loop(self.train_loader, self.model, self.loss_fn, self.optimizer,
                                                    self.device, self.hyp.get("decay_rate", None))
            val_loss, accuracy = valid_loop(self.valid_loader, self.model, self.loss_fn, self.device)

            # Loss tracking
            self.train_tracer.track(tt_loss, "total")
            self.train_tracer.track(md_loss, "model")
            self.train_tracer.track(reg_loss, "regula")
            self.valid_tracer.track(val_loss)
            self.acc_tracer.track(accuracy)

            if self.on_epoch_end(valid_loss=val_loss, accuracy=accuracy):    # Epoch end
                break

        self.on_train_end()  # Training end

    def model_save(self, model_path, model_prefix="yolov5"):
        # Saving to model/plan_*/yolov5_mmdd-HHMM.pth
        stamp = time.strftime("%m%d-%H%M%S")
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        # Model saved with OrderDict using torch
        model_fp = join(model_path, "%s_%s.pth" % (model_prefix, stamp))
        torch.save(self.model, model_fp)

        # Hyper-params saving using yaml
        dump_dict = {"hyp":         self.hyp,
                     "max_epochs":  self.stop_epoch,
                     "train_loss":  self.train_tracer.boxes["total"].box,
                     "model_loss":  self.train_tracer.boxes["model"].box,
                     "regula_loss":  self.train_tracer.boxes["regula"].box,
                     "valid_loss":  self.valid_tracer.box,
                     "accuracy":    self.acc_tracer.box}

        losses_fpath = join(model_path, "losses_%s.yaml" % stamp)
        with open(losses_fpath, "w") as loss_f:
            yaml.dump(dump_dict, loss_f)
        print(f"Model weights saved in '{abspath(model_fp)}'\n")


if __name__ == '__main__':
    print(f"Torch Version: {torch.__version__}, Cuda Available: {torch.cuda.is_available()}")
    # Logging Config ---------------------------------------------------
    logging.config.fileConfig(".\\log\\config\\train.conf")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Logging Config Ended ---------------------------------------------

    # ============================ Model Training ==================================
    try:
        trainer = Trainer(".\\hypers.yaml")
        trainer.training(100)
    except Exception as e:
        logger.exception(e)
