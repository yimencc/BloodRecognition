import os
import time
import logging
from logging import config
from collections import ChainMap
from os.path import join, abspath

import yaml
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Deeplearning.evaluate import Set2021091Metadata
from Deeplearning.util.models import YoloV6Model
from Deeplearning.util.dataset import create_dataloader, ANCHORS, GRIDSZ
from Deeplearning.util.losses import YoloLoss, TracerMini, Tracer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Meta-Parameters
CLASS_LOC = 0
N_CLASSES = 3
DST_IMGSZ = 320
SEED = 123456
IMG_PLUGIN = "simpleitk"
MODEL_PATH = "..\\models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed the random states
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


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
            equation = self.best_acc - val_acc - self.min_delta
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
        x, y = modalities.to(device), [item.to(device) for item in labels]
        # Forward propagation
        pred = model(x)
        # Compute training loss
        reg_loss = decay_rate * sum([torch.square(torch.sum(param.data)) for param in model.parameters()])
        total_loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Batch loss
        n_current += len(x)
        cur_loss = total_loss.item()
        mod_loss = cur_loss
        reg_loss = reg_loss.item()

        # Epoch loss update
        train_loss += cur_loss * len(x)
        model_loss += cur_loss * len(x)
        regula_loss += reg_loss * len(x)

        if (batch + 1) % 10 == 0:
            print(f"Total: {cur_loss:>7f}   Model: {mod_loss:>7f}   "
                  f"Regula: {reg_loss:>7f}   [{n_current:>5d}/{size:>5d}]")

    avg_train_loss = train_loss / size
    avg_model_loss = model_loss / size
    avg_regula_loss = regula_loss / size
    print(f"Train Error: [Avg Loss {avg_train_loss:>8f}, Avg Model {avg_model_loss:>8f}, "
          f"Avg Regula {avg_regula_loss:>8f}]")
    return avg_train_loss, avg_model_loss, avg_regula_loss


def valid_loop(dataloader, model, loss_fn, device):
    print("\nTest\n" + "-" * 30)
    test_loss, accuracy, n_current, size = 0, 0, 0, len(dataloader.dataset)
    with torch.no_grad():
        for batch, (modalities, labels) in enumerate(dataloader):
            # Transfer data to gpu
            x, y = modalities.to(device), [item.to(device) for item in labels]
            # Forward propagation
            pred = model(x)
            # Compute batch test loss
            cur_loos = loss_fn(pred, y).item()
            cur_acy = loss_fn.accuracy.item()

            # Epoch data
            n_current += len(x)
            test_loss += cur_loos * len(x)
            accuracy += cur_acy * len(x)
            if (batch + 1) % 5 == 0:
                print(f"loss: {cur_loos:>7f}  accuracy: {cur_acy:>5f}  [{n_current:>5d}/{size:>5d}]")

    test_loss /= size
    accuracy /= size
    print(f"Test Error: [Avg loss {test_loss:>8f}, Avg accuracy {accuracy:>3f}]\n")
    return test_loss, accuracy


class Trainer:
    def __init__(self, hyp, train_loader=None, valid_loader=None, **kwargs):
        if isinstance(hyp, str) and hyp.endswith(".yaml"):
            with open(hyp, "r") as hyp_file:
                hyp_dict = yaml.load(hyp_file, yaml.FullLoader)
        self.hyp = ChainMap(kwargs, hyp_dict)

        params_names = ("n_box", "n_cls", "input_channel", "grid_size", "anchors", "attention_layer", "lr")
        n_box, n_cls, input_channel, grid_sz, anchors, attention, lr = [self.hyp.get(item) for item in params_names]
        self.device = self.hyp.get("device", DEVICE)
        self.input_shape = (input_channel, 320, 320)
        self.stop_epoch = 0
        self.callbacks = {}
        self.train_break = False
        self.model_filename = None

        self.loss_fn = YoloLoss(self.device, anchors, grid_sz, n_cls)
        self.model = YoloV6Model(input_channel, n_box=n_box, n_cls=n_cls, grid_size=grid_sz).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.train_tracer = Tracer("train")
        self.valid_tracer = TracerMini("valid")
        self.acc_tracer = TracerMini("acc")
        self.train_tracer.register_boxes(("total", "model", "regula"))

        # Register Default Callback
        dft_callback = {"ReduceLR": ReduceLROnPlateau, "EarlyStopping": EarlyStopping}
        self.register_callbacks(schedule=(dft_callback["ReduceLR"], {"optimizer": self.optimizer, "mode": "min"}),
                                early_stopping=(dft_callback["EarlyStopping"], {"patience": 10}))
        # Load data
        logger.info("Loading data ...")
        if not train_loader:
            train_loader = create_dataloader(**self.hyp.get("train_loader_params"))
        if not valid_loader:
            valid_loader = create_dataloader(**self.hyp.get("valid_loader_params"))
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def register_callbacks(self, **callbacks):
        """ callbacks: {"cb_name": (callback, params_dict)} """
        for cb_name, (callback, param_dict) in callbacks.items():
            assert isinstance(cb_name, str)
            assert isinstance(param_dict, dict)
            self.callbacks.update({cb_name: callback(**param_dict)})

    def on_train_begin(self, **kwargs):
        # Hyper-params
        print("\nHyper-parameters:\n" + "=" * 30)
        for k, v in self.hyp.items():
            print(f"\t{k}: {v}")

        # Model Info
        if kwargs.get("model_info", None):
            print("\nModel Info")
            from torchsummary import summary
            summary(self.model, self.input_shape)
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

            if self.on_epoch_end(valid_loss=val_loss, accuracy=accuracy):  # Epoch end
                break

        self.on_train_end()  # Training end
        return self.model_filename

    def model_save(self, model_path, model_prefix="yolov5"):
        # Saving to model/plan_*/yolov5_mmdd-HHMM.pth
        stamp = time.strftime("%m%d-%H%M%S")
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        # Model saved with OrderDict using torch
        self.model_filename = join(model_path, "%s_%s.pth" % (model_prefix, stamp))
        torch.save(self.model, self.model_filename)

        # Hyper-params saving using yaml
        dump_dict = {
            "hyp": self.hyp,
            "max_epochs": self.stop_epoch,
            "accuracy": self.acc_tracer.box,
            "valid_loss": self.valid_tracer.box,
            "train_loss": self.train_tracer.boxes["total"].box,
            "model_loss": self.train_tracer.boxes["model"].box,
            "regula_loss": self.train_tracer.boxes["regula"].box,
        }

        losses_fpath = join(model_path, "losses_%s.yaml" % stamp)
        with open(losses_fpath, "w") as loss_f:
            yaml.dump(dump_dict, loss_f)
        print(f"Model weights saved in '{abspath(self.model_filename)}'\n")


if __name__ == '__main__':
    print(f"Torch Version: {torch.__version__}, Cuda Available: {torch.cuda.is_available()}")
    # Logging Config ---------------------------------------------------
    logging.config.fileConfig(".\\log\\config\\train.conf")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Logging Config Ended ---------------------------------------------

    # ============================ Model Training ==================================
    try:
        modelFpath = join(MODEL_PATH, "plan_8.4")
        trainer = Trainer(".\\train_hypers.yaml", model_fpath=modelFpath)
        modelFilename = trainer.training(100)

        # evaluating
        accSrcFilename = join(modelFpath, "acc_compare_set202109-1.pkl")
        set2021091 = Set2021091Metadata(source_filename=accSrcFilename, refresh=True)

        predParams = {"image_shape": (DST_IMGSZ, DST_IMGSZ), "grid_size": GRIDSZ,
                      "anchors": ANCHORS, "n_class": N_CLASSES, "device": "cpu",
                      "nms_overlapping_thres": .3, "conf_thres": .6}
        set2021091.auto_verse_dl(modelFilename, predParams)
    except Exception as e:
        logger.exception(e)
