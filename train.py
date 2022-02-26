import os
import time
from collections import ChainMap
from os.path import join, abspath, exists

import yaml
import torch
from torch.optim.lr_scheduler import MultiStepLR

from Deeplearning.util.losses import YoloLoss
from Deeplearning.util.models import YoloV6Model
from Deeplearning.util.dataset import create_dataloader

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Meta-Parameters
SEED = 123456
IMG_PLUGIN = "simpleitk"
MODEL_PATH = "..\\models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Fixed the random states
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.autograd.set_detect_anomaly(True)


class LossTracer:
    def __init__(self, name: str):
        self.name = name
        self.box = []
        self.counter = 0

    def track(self, value, c=None):
        self.box.append(value)
        if c is not None:
            self.counter += c

    def update_state(self):
        self.box = []

    def __getitem__(self, item):
        return self.box[item]


class EarlyStopping:
    """ Early stopping to stop the training when the loss does not improve after
    certain epochs. """
    def __init__(self, patience=5, min_delta=0, mode: str = "max"):
        """
        param patience: how many epochs to wait before stopping when loss is
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


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    train_loss, n_trained = 0, 0
    for batch, (modalities, labels) in enumerate(dataloader):
        # Transfer data to gpu
        x, y = modalities.to(device), [item.to(device) for item in labels]
        # Forward propagation
        pred = model(x)
        # Compute training loss
        loss, (lcoord, lobj, lnonobj, lcls) = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Batch loss
        n_trained += len(x)
        train_loss += loss.item() * len(x)
        if (batch + 1) % 10 == 0:
            print(f"Loss: {loss.item():>.5f} (coord: {lcoord:>.4f}, obj: {lobj:>.4f}, "
                  f"non_obj: {lnonobj:>.4f}, class: {lcls:>.4f}) [{n_trained:>5d}/{size:>5d}]")

    avg_train_loss = train_loss / size
    print(f"Train Avg Loss {avg_train_loss:>8f}")
    return avg_train_loss


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
            loss, _ = loss_fn(pred, y)
            cur_acc = loss_fn.accuracy.item()
            # Epoch data
            n_current += len(x)
            test_loss += loss.item() * len(x)
            accuracy += cur_acc * len(x)
            if (batch + 1) % 5 == 0:
                print(f"loss: {loss.item():>.5f}  accuracy: {cur_acc:>.5f}  [{n_current:>5d}/{size:>5d}]")

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
        names = ("n_box", "n_cls", "input_channel", "grid_size", "anchors", "attention_layer", "lr")
        n_box, n_cls, input_channel, grid_sz, anchors, attention, lr = [self.hyp.get(item) for item in names]
        self.device = self.hyp.get("device", DEVICE)
        self.input_shape = (input_channel, 320, 320)
        self.model_filename = None
        self.train_break = False
        self.stop_epoch = 0
        self.callbacks = {}

        self.loss_fn = YoloLoss(self.device, anchors, grid_sz, n_cls)
        self.model = YoloV6Model(input_channel, n_box=n_box, n_cls=n_cls, grid_size=grid_sz).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.train_tracer = LossTracer("train")
        self.valid_tracer = LossTracer("valid")
        self.acc_tracer = LossTracer("acc")

        # Register Default Callback
        schedule_params = {"optimizer": self.optimizer, "milestones": [20, 50], "gamma": 0.1}
        self.register_callbacks(early_stopping=(EarlyStopping, {"patience": 7}),
                                schedule=(MultiStepLR, schedule_params))
        # Load data
        print("Loading data ...")
        self.train_loader = create_dataloader(**self.hyp.get("train_loader")) if not train_loader else train_loader
        self.valid_loader = create_dataloader(**self.hyp.get("valid_loader")) if not valid_loader else valid_loader

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
        print("Training Start")

    def on_train_end(self, **kwargs):
        # Model Save
        if model_fpath := self.hyp.get("model_fpath", None):
            model_prefix = kwargs.get("model_prefix", "yolov6")
            self.model_save(model_fpath, model_prefix)

    def on_epoch_begin(self, **kwargs):
        # Epoch Message
        lr = self.callbacks.get("schedule", None).get_last_lr()[0]
        if epoch := kwargs.get("epoch", None):
            print(f"[Epoch {epoch}, lr: {lr}]\n" + "-" * 30)

    def on_epoch_end(self, **kwargs):
        """ Mainly for call back executing """
        self.stop_epoch += 1
        # Learning Rate Schedule
        if schedule := self.callbacks.get("schedule", None):
            schedule.step()

        # Early Stopping
        if early_stopping := self.callbacks.get("early_stopping"):
            accuracy = kwargs.get("accuracy")
            early_stopping.step(accuracy)
            if early_stopping.early_stop:
                print(f"Early stopped at Accuracy {accuracy:.4f}")
                return True

    def training(self, epochs):
        self.on_train_begin(model_info=True)
        for t in range(epochs):
            self.on_epoch_begin(epoch=t + 1)
            # Train and Valid
            tra_loss = train_loop(self.train_loader, self.model, self.loss_fn, self.optimizer, self.device)
            val_loss, accuracy = valid_loop(self.valid_loader, self.model, self.loss_fn, self.device)

            # Loss tracking
            self.train_tracer.track(tra_loss)
            self.valid_tracer.track(val_loss)
            self.acc_tracer.track(accuracy)
            if self.on_epoch_end(valid_loss=val_loss, accuracy=accuracy):  # Epoch end
                break

        self.on_train_end()  # Training end
        return self.model_filename

    def model_save(self, model_path, model_prefix="yolov6"):
        # Saving to model/plan_*/yolov5_mmdd-HHMM.pth
        stamp = time.strftime("%m%d-%H%M%S")
        if not exists(model_path):
            os.mkdir(model_path)
        # Model saved with OrderDict using torch
        self.model_filename = join(model_path, "%s_%s.pth" % (model_prefix, stamp))
        torch.save(self.model, self.model_filename)
        print(f"Model weights saved in '{abspath(self.model_filename)}'\n")

        # Hyper-params saving using yaml
        dump_dict = {
            "hyp": dict(self.hyp),
            "max_epochs": self.stop_epoch,
            "accuracy": self.acc_tracer.box,
            "valid_loss": self.valid_tracer.box,
            "train_loss": self.train_tracer.box,
        }
        losses_fpath = join(model_path, "losses_%s.yaml" % stamp)
        with open(losses_fpath, "w") as loss_f:
            yaml.dump(dump_dict, loss_f)
