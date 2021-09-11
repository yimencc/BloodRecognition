import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
from os.path import join

import torch
import numpy as np
from torch.utils.data import DataLoader

from models     import YoloV5Model
from dataset    import RBCXmlDataset, TRAIN_DS_CONSTRUCTOR, VALID_DS_CONSTRUCTOR

print("Torch Version: ",    torch.__version__)
print("Cuda Available: ",   torch.cuda.is_available())

IMGSZ       =   320
IMG_PLUGIN  =   "simpleitk"
MODEL_PATH  =   "..\\data\\models"
device      =   "cuda" if torch.cuda.is_available() else "cpu"
seed        =   123456

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class _Loss:
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


def train_loop(dataloader, model, loss_fn, optimizer, decay_rate=.01):
    size = len(dataloader.dataset)
    train_loss = []
    for batch, db_train in enumerate(dataloader):
        # Compute prediction and loss
        X, y = db_train["modality"].to(device), db_train["label"]
        y = [item.to(device) for item in y]
        pred = model(X)
        regula_loss = 0
        for param in model.parameters():
            regula_loss += torch.sum(param.data)
        loss = loss_fn(pred, y)
        total_loss = loss + decay_rate * regula_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
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
            X, y = db_valid["modality"].to(device), db_valid["label"]
            y = [item.to(device) for item in y]
            pred = model(X)
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


class TrainingPlan:
    """Search the best parameters for model training"""
    def __init__(self, name, epochs, batch_size, train_set, valid_set,
                 callbacks=None, cb_params=None, model_fname=None, storage_mode="dict"):
        self.model_path = os.path.join(MODEL_PATH, name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.logs = {}
        self.model = None
        self.epochs = epochs
        self.callbacks = callbacks
        self.cb_params = cb_params
        self.batch_size = batch_size
        self.model_fname = model_fname
        self.storage_mode = storage_mode
        self.train_loader = DataLoader(train_set, batch_size, True)
        self.valid_loader = DataLoader(valid_set, batch_size, True)

        self.accuracy = _Loss("acc")
        self.train_losses = _Loss("train")
        self.valid_losses = _Loss("valid")

    def training(self, learning_rate, decay_rate):
        loss_fn = losses.YoloLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        schedule, early_stopping = None, None
        if self.callbacks == "lr_schedule":
            schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(**self.cb_params)
        elif self.callbacks == "early_stopping":
            early_stopping = EarlyStopping(**self.cb_params)

        max_epochs = 0
        self.train_losses = _Loss("train")
        self.valid_losses = _Loss("valid")
        self.accuracy = _Loss("acc")
        for t in range(self.epochs):
            print(f"Epoch {t + 1}\n" + "-" * 31)
            self.train_losses.track(train_loop(self.train_loader, self.model,
                                               loss_fn, optimizer, decay_rate))
            valid_loss, acc = valid_loop(self.valid_loader, self.model, loss_fn)
            self.valid_losses.track(valid_loss)
            self.accuracy.track(acc)
            max_epochs += 1

            if self.callbacks == "lr_schedule":
                schedule.step(valid_loss)
            elif self.callbacks == "early_stopping":
                early_stopping(acc)
                if early_stopping.early_stop:
                    print("Early stopped")
                    break

        self.model_save(learning_rate, decay_rate, max_epochs)

    def model_save(self, learning_rate, decay_rate, max_epochs):
        # Saving -> data/models/model_weights_mmdd-HHMM.pth
        t_stamp = time.strftime("%m%d-%H%M%S")
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        model_fp = join(self.model_path, "yoloV2_%s.pth" % t_stamp)
        if self.storage_mode == "dict":
            torch.save(self.model.state_dict(), model_fp)
        elif self.storage_mode == "pickle":
            torch.save(self.model, model_fp)

        losses_fp = join(self.model_path, "losses_%s.txt" % t_stamp)
        with open(losses_fp, "w") as loss_f:
            loss_f.write(f"Initial lr:{learning_rate}\n"
                         f"Decay_rate:{decay_rate}\n"
                         f"Max Epochs:{max_epochs}\n"
                         f"Train Losses\n")
            for loss in self.train_losses.box:
                loss_f.write(str(loss) + "\n")
            loss_f.write("Valid Losses:\n")
            for loss in self.valid_losses.box:
                loss_f.write(str(loss) + "\n")
            loss_f.write("Accuracy:\n")
            for acc in self.accuracy.box:
                loss_f.write(str(acc) + "\n")
        print(f"Model weights saved in {os.path.abspath(model_fp)}\n")

    def execute(self, learning_rates, decay_rates, patience):
        """ Execute the given plans """

        self.logs = {"epochs": self.epochs,
                     "batch_size": self.batch_size,
                     "learning_rates": learning_rates,
                     "decay_rates": decay_rates,
                     "patience": patience}

        best_params = {}
        best_performance = 0
        for i, lr in enumerate(learning_rates):
            for k, decay_rate in enumerate(decay_rates):
                cur_stage = i * len(decay_rates) + k + 1
                print("stage {:d} lr: {:.5f} dc_rate: {:.4f}, patience: {:d}\n{:s}"
                      .format(cur_stage, lr, decay_rate, patience, "="*50))
                self.model = YoloV5Model(attention_layer=7)
                self.model = self.model.to(device)
                # self.model.initialize_weights()
                if self.model_fname:
                    print(f"Load Model from: {self.model_fname}")
                    if self.storage_mode == "dict":
                        self.model.load_state_dict(torch.load(self.model_fname))
                    elif self.storage_mode == "pickle":
                        self.model = torch.load(self.model_fname)
                self.training(lr, decay_rate)
                if (i == 0 and k == 0) or self.accuracy[-1] > best_performance:
                    best_performance = self.accuracy[-1]
                    best_params.update({"learning_rate": lr,
                                        "decay_rate": decay_rate,
                                        "patience": patience})
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


def main():
    PLAN_DICT = {"epochs": 100,
                 "batch_size": 8,
                 "storage_mode": "dict",
                 "train_set": RBCXmlDataset(**TRAIN_DS_CONSTRUCTOR),
                 "valid_set": RBCXmlDataset(**VALID_DS_CONSTRUCTOR),
                 "callbacks": "early_stopping",
                 "cb_params": {"patience": 7}}

    train_plan = TrainingPlan(name="plan_7.0", **PLAN_DICT)
    train_plan.execute(learning_rates=[3e-4], decay_rates=[0.03], patience=7)


if __name__ == '__main__':
    main()
