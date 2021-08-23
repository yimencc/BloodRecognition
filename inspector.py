from os.path import join

import torch
from torch.nn import Module

from models import YoloV2Model
from dataset import RbcDataset, DataLoader, VALID_DS_CONSTRUCTOR


class Inspector:
    def __init__(self, model, dataloader):
        self.model: Module = model
        self.dataloader = dataloader

    def forward(self):
        sample = next(iter(self.dataloader))
        image, label = sample["image"], sample["label"]

    def get_layer(self):
        pass


def model_layers_inspect():
    pass


def main():
    plan_name = "plan_4.5"
    model_path = "..\\data\\models"
    model_name = "yolov2_0415-134352.pth"

    yolo_model = YoloV2Model()
    cur_model_fname = join(model_path, plan_name, model_name)
    yolo_model.load_state_dict(torch.load(cur_model_fname))

    rbcDataset = RbcDataset(**VALID_DS_CONSTRUCTOR)
    valid_dataloader = DataLoader(rbcDataset, batch_size=1, shuffle=False)


if __name__ == '__main__':
    main()
