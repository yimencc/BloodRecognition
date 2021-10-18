import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from Deeplearning.util.models import YoloV5Model


class HyperSearchingPlan:
    """
    Search the best parameters for model training, suitable for quickly
    finding super parameters after the model training process is stable.
    Notice: It is not recommended to use it when there are problems in the training process!!!
    TODO: should be divided into two class_scores: Plan(For hyperParam searching) and Training(For training implementing)
    TODO: For every group of parameters, save the trained model and parameters
    """
    def __init__(self, name, epochs, batch_size, train_set, valid_set, device,
                 model_path, model_fname=None, store_mode="dict", loss_tracer=None):

        self.model_path = os.path.join(model_path, name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.logs           =   {}
        self.model          =   None
        self.epochs         =   epochs
        self.max_epochs     =   0
        self.device         =   device

        # TODO: Achieve 'callbacks' and 'cb_params' through registering_callbacks
        self.callbacks      =   OrderedDict()
        self.cb_params      =   OrderedDict()

        self.batch_size     =   batch_size
        self.model_fname    =   model_fname
        self.store_mode     =   store_mode

        # TODO: get dataloader through create_dataloader from dataset module
        self.train_loader   =   DataLoader(train_set, batch_size, True)
        self.valid_loader   =   DataLoader(valid_set, batch_size, True)

        self.accuracy_tracer    =   loss_tracer("acc")
        self.train_ls_tracer    =   loss_tracer("train")
        self.valid_ls_tracer    =   loss_tracer("valid")

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
                self.model = self.model.to(self.device)
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
    # ====================== Hyper-parameters Searching ============================
    try:
        train_plan = HyperSearchingPlan(name="plan_8.0",
                                        epochs=2,
                                        batch_size=8,
                                        store_mode="dict",
                                        model_path=MODEL_PATH,
                                        train_set=BloodSmearDataset.from_xml_cache(**TRAIN_DS_CACHES),
                                        valid_set=BloodSmearDataset.from_xml_cache(**TRAIN_DS_CACHES))

        train_plan.search(learning_rates=[3e-4], decay_rates=[0.03], patience=7)

    except Exception as e:
        logger.exception(e)     # Error logging
