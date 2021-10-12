import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import logging
from logging import config
from os.path import join
from collections import OrderedDict

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from Deeplearning.util.losses   import YoloLoss
from Deeplearning.util.models   import YoloV5Model
from Deeplearning.util.dataset  import BloodSmearDataset, ANCHORS, GRIDSZ, TRAIN_DS_CACHES

# Meta-Parameters
IMGSZ       =   320
seed        =   123456
IMG_PLUGIN  =   "simpleitk"
MODEL_PATH  =   "..\\models"
DEVICE      =   "cuda" if torch.cuda.is_available() else "cpu"
CLASS_LOC   =   0
N_CLASSES   =   3

# Fixed the random states
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Logging Config ---------------------------------------------------
logging.config.fileConfig(".\\log\\config\\train.conf")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Logging Config Ended ---------------------------------------------


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


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
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

    def __call__(self, val_acc):
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


def train_loop(dataloader, model, loss_fn, optimizer, decay_rate=.01, regularization=False):
    train_loss = []
    size = len(dataloader.dataset)
    for batch, db_train in enumerate(dataloader):
        # Transfer data to gpu
        X, y = db_train["modalities"].to(DEVICE), [item.to(DEVICE) for item in db_train["labels"]]

        # Forward propagation
        pred = model(X)

        # Compute training loss
        loss = loss_fn(pred, y)
        if regularization:
            regula_loss =   sum([torch.sum(param.data) for param in model.parameters()])
            loss        +=  decay_rate * regula_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item()*len(X))
        if (batch+1) % 5 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    epoch_avg_loss = sum(train_loss)/size
    print(f"Train Error: Avg Loss {epoch_avg_loss:>8f}")
    return epoch_avg_loss


def valid_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, accuracy = 0, 0

    print("\nTest:")
    with torch.no_grad():
        for batch, db_valid in enumerate(dataloader):
            # Transfer data to gpu
            X, y = db_valid["modalities"].to(DEVICE), [item.to(DEVICE) for item in db_valid["labels"]]

            # Forward propagation
            pred = model(X)

            # Compute test loss
            cur_loos = loss_fn(pred, y).item()
            test_loss += cur_loos*len(X)
            cur_acy = loss_fn.accuracy.item()
            accuracy += cur_acy * len(X)

            if (batch+1) % 5 == 0:
                loss, current = cur_loos, (batch+1)*len(X)
                print(f"loss: {loss:>7f}  accuracy: {cur_acy:>5f}  [{current:>5d}/{size:>5d}]")

    test_loss /= size
    accuracy /= size
    print(f"Test Error: Avg loss {test_loss:>8f}, Avg accuracy {accuracy:>3f}\n")
    return test_loss, accuracy


class Trainer:
    def __init__(self, hyp):
        if isinstance(hyp, str) and hyp.endswith(".yaml"):
            with open(hyp, "r") as hyp_f:
                self.hyp = yaml.load(hyp_f, yaml.FullLoader)

        params_list         =   ("lr", "n_box", "n_cls", "grid_size", "attention_layer", "anchors")
        lr, n_box, n_cls, grid_sz, attention_ly, anchors = [self.hyp.get(item) for item in params_list]

        self.max_epochs     =   0
        self.callbacks      =   {}

        self.loss_fn        =   YoloLoss(DEVICE, anchors, grid_sz, n_cls)
        self.model          =   YoloV5Model(n_box, n_cls, grid_sz, attention_ly).to(DEVICE)
        self.optimizer      =   torch.optim.Adam(self.model.parameters(), lr=self.hyp["lr"])

        self.acc_tracer     =   _LossTracer("acc")
        self.train_tracer   =   _LossTracer("train")
        self.valid_tracer   =   _LossTracer("valid")

    def registering_training_callbacks(self, names):
        for name in names:
            if name == "lr_schedule":
                schedule_dict = {}
                self.callbacks.update(
                    {"schedule": torch.optim.lr_scheduler.ReduceLROnPlateau(**schedule_dict)}
                )
            if name == "early_stopping":
                early_stopping_dict = {}
                self.callbacks.update(
                    {"early_stopping": EarlyStopping(**early_stopping_dict)}
                )

    def before_training(self):
        pass

    def after_epoch(self, **kwargs):
        """
        Mainly for call back executing
        """
        self.max_epochs += 1

        # Callback search
        valid_loss  =   kwargs.get("valid_loss")
        acc         =   kwargs.get("acc")

        schedule = self.callbacks.get("lr_schedule", None)
        if schedule:
            schedule.step(valid_loss)

        early_stopping = self.callbacks.get("early_stopping")
        if early_stopping:
            early_stopping(acc)
            if early_stopping.early_stop:
                print("Early stopped")
                pass

    def training(self, epochs, train_loader, valid_loader):
        self.before_training()

        decay_rate = self.hyp["decay_rate"] if "decay_rate" in self.hyp.keys() else None
        for t in range(epochs):
            print(f"Epoch {t+1}\n" + "-" * 30)
            # Track training
            train_loss = train_loop(train_loader, self.model, self.loss_fn, self.optimizer, decay_rate)

            # Track testing
            valid_loss, acc = valid_loop(valid_loader, self.model, self.loss_fn)

            self.train_tracer.track(train_loss)
            self.valid_tracer.track(valid_loss)
            self.acc_tracer.track(acc)

            self.after_epoch()

    def model_save(self, model_path, model_prefix="yolov5"):
        # Saving to model/plan_*/yolov5_mmdd-HHMM.pth
        t_stamp = time.strftime("%m%d-%H%M%S")
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        # Model saved with OrderDict using torch
        model_fp = join(model_path, "%s_%s.pth" % (model_prefix, t_stamp))
        torch.save(self.model.state_dict(), model_fp)

        # Hyper-params saving using yaml
        dump_dict = {"hyp":         self.hyp,
                     "max_epochs":  self.max_epochs,
                     "train_loss":  self.train_tracer.box,
                     "valid_loss":  self.valid_tracer.box,
                     "accuracy":    self.acc_tracer.box}

        losses_fp = join(model_path, "losses_%s.yaml" % t_stamp)
        with open(losses_fp, "w") as loss_f:
            yaml.dump(dump_dict, loss_f)

        print(f"Model weights saved in {os.path.abspath(model_fp)}\n")


class HyperSearchingPlan:
    """
    Search the best parameters for model training, suitable for quickly
    finding super parameters after the model training process is stable.
    Notice: It is not recommended to use it when there are problems in the training process!!!
    TODO: should be divided into two classes: Plan(For hyperParam searching) and Training(For training implementing)
    TODO: For every group of parameters, save the trained model and parameters
    """
    def __init__(self, name, epochs, batch_size, train_set, valid_set,
                 model_path, model_fname=None, store_mode="dict"):

        self.model_path = os.path.join(model_path, name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.logs           =   {}
        self.model          =   None
        self.epochs         =   epochs
        self.max_epochs     =   0

        # TODO: Achieve 'callbacks' and 'cb_params' through registering_training_callbacks
        self.callbacks      =   OrderedDict()
        self.cb_params      =   OrderedDict()

        self.batch_size     =   batch_size
        self.model_fname    =   model_fname
        self.store_mode     =   store_mode

        # TODO: get dataloader through create_dataloader from dataset module
        self.train_loader   =   DataLoader(train_set, batch_size, True)
        self.valid_loader   =   DataLoader(valid_set, batch_size, True)

        self.accuracy_tracer    =   _LossTracer("acc")
        self.train_ls_tracer    =   _LossTracer("train")
        self.valid_ls_tracer    =   _LossTracer("valid")

    def search(self, learning_rates, decay_rates, patience):
        """ Execute the given plans """

        self.logs = {"epochs":          self.epochs,
                     "batch_size":      self.batch_size,
                     "learning_rates":  learning_rates,
                     "decay_rates":     decay_rates,
                     "patience":        patience}

        best_params = {}
        best_performance = 0
        for i, lr in enumerate(learning_rates):
            for k, decay_rate in enumerate(decay_rates):
                cur_stage = i * len(decay_rates) + k + 1
                print("stage {:d} lr: {:.5f} dc_rate: {:.4f}, patience: {:d}\n{:s}"
                      .format(cur_stage, lr, decay_rate, patience, "="*50))

                # Model initialization
                self.model = YoloV5Model(attention_layer=7)
                self.model = self.model.to(DEVICE)
                # self.model.initialize_weights()

                # Read weights from disk
                if self.model_fname:
                    print(f"Load Model from: {self.model_fname}")
                    if self.store_mode == "dict":
                        self.model.load_state_dict(torch.load(self.model_fname))
                    elif self.store_mode == "pickle":
                        self.model = torch.load(self.model_fname)

                    # Implementing model training

                # Update best stages and parameters
                if (i == 0 and k == 0) or self.accuracy_tracer[-1] > best_performance:
                    best_performance = self.accuracy_tracer[-1]
                    best_params.update({"learning_rate": lr, "decay_rate": decay_rate, "patience": patience})

        # Save logs. Training Ended!
        self.msg_log(best_params)

    def msg_log(self, best_params):
        log_name = os.path.join(self.model_path, "log.txt")
        with open(log_name, "w") as f:
            if self.model_fname:
                f.write("model load from: %s\n" % self.model_fname)
            for key, val in self.logs.items():
                f.write("%s: %s\n" % (key, str(val)))
            f.write("Best Performance:\n")
            for key, val in best_params.items():
                f.write("%s: %s\n" % (key, str(val)))


if __name__ == '__main__':
    print(f"Torch Version: {torch.__version__}, Cuda Available: {torch.cuda.is_available()}")

    # ============================ Model Training ==================================
    hyper_cfg   =   ".\\hypers.yaml"
    modelSpath  =   join(MODEL_PATH, "plan_8.0")
    dataLoader  =   DataLoader(BloodSmearDataset.from_cache(**TRAIN_DS_CACHES), batch_size=8)

    trainer     =   Trainer(hyper_cfg)
    trainer.training(2, dataLoader, dataLoader)
    trainer.model_save(modelSpath)

    # # ====================== Hyper-parameters Searching ============================
    # try:
    #     train_plan = HyperSearchingPlan(name="plan_8.0",
    #                                     epochs=2,
    #                                     batch_size=8,
    #                                     store_mode="dict",
    #                                     model_path=MODEL_PATH,
    #                                     train_set=BloodSmearDataset.from_cache(**TRAIN_DS_CACHES),
    #                                     valid_set=BloodSmearDataset.from_cache(**TRAIN_DS_CACHES))
    #
    #     train_plan.search(learning_rates=[3e-4], decay_rates=[0.03], patience=7)
    #
    # except Exception as e:
    #     logger.exception(e)     # Error logging
