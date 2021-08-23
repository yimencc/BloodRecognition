import math

import torch
import torch.nn as nn

from dataset import GRIDSZ


def load_model():
    return None


class CAB(nn.Module):
    # channel attention
    def __init__(self, in_features):
        super(CAB, self).__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_features, in_features // 4, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_features // 4, in_features, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        poolout = self.fc2(self.relu(self.fc1(self.maxpooling(x))))
        return self.sigmoid(poolout)


class SAB(nn.Module):
    # spatial attention
    def __init__(self, kernel_size=3):
        super(SAB, self).__init__()
        assert kernel_size in (3, 7)  # 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        self.block_size = block_size
        super(SpaceToDepth, self).__init__()

    def forward(self, x):
        batch, depth, height, width = x.shape
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = x.view(batch, depth, self.block_size, reduced_height, self.block_size, reduced_width)
        z = y.permute(0, 1, 2, 4, 3, 5)
        t = z.reshape(batch, depth*self.block_size**2, reduced_height, reduced_width)
        return t

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0], input_shape[1] // self.block_size,
                 input_shape[2] // self.block_size,
                 input_shape[3] * self.block_size ** 2)
        return shape


class ConvBlock(nn.Module):
    def __init__(self, n_filters, raised_tail=False):
        super(ConvBlock, self).__init__()
        self.n_filters = n_filters

        # layers define
        self.lk_relu = nn.LeakyReLU(0.1)

        if not raised_tail:
            self.conv_3 = nn.Conv2d(self.n_filters, self.n_filters, (3, 3), stride=(1, 1),
                                    padding=1, padding_mode='zeros', bias=False)
            self.bn_3 = nn.BatchNorm2d(self.n_filters)
        else:
            self.conv_3 = nn.Conv2d(self.n_filters, 2*self.n_filters, (3, 3), stride=(1, 1),
                                    padding=1, padding_mode='zeros', bias=False)
            self.bn_3 = nn.BatchNorm2d(2*self.n_filters)

    def forward(self, x):
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.lk_relu(x)
        return x


# class YoloV3Model(nn.Module):
#     def __init__(self, n_box=4, n_cls=4):
#         super(YoloV3Model, self).__init__()
#         self.gridsz = GRIDSZ
#         self.n_box = n_box
#         self.n_cls = n_cls
#
#         # layers define
#         self.lk_relu = nn.LeakyReLU(0.1)
#         self.pool = nn.MaxPool2d((2, 2))
#
#         # unit 1  (b, 1, 320, 320) -> (b, 16, 160, 160)
#         self.conv_1 = nn.Conv2d(1, 16, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_1 = nn.BatchNorm2d(16)
#
#         # unit 2  (b, 16, 160, 160) -> (b, 32, 160, 160)
#         self.conv_2 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_2 = nn.BatchNorm2d(32)
#
#         # unit 3  (b, 32, 160, 160) -> (b, 64, 80, 80)
#         self.conv_3 = nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_3 = nn.BatchNorm2d(64)
#
#         # unit 4  (b, 64, 80, 80) -> (b, 64, 80, 80)
#         self.conv_4 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_4 = nn.BatchNorm2d(64)
#
#         # unit 5  max_pooling -> (b, 40, 40, 128)
#         self.conv_5 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_5 = nn.BatchNorm2d(128)
#
#         # unit 6  (b, 128, 40, 40) -> (b, 256, 40, 40)
#         self.conv_6 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_6 = nn.BatchNorm2d(256)
#
#         # unit 7  Skip_x block (b, 64, 80, 80) -> (b, 16, 80, 80)
#         self.conv_7 = nn.Conv2d(64, 16, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_7 = nn.BatchNorm2d(16)
#
#         # unit 8  (b, 320, 40, 40) -> (b, 256, 40, 40)
#         self.conv_8 = nn.Conv2d(320, 256, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_8 = nn.BatchNorm2d(256)
#
#         # unit 7  (b, 256, 40, 40) -> (b, 4*9, 40, 40)
#         self.conv_9 = nn.Conv2d(256, self.n_box*(5+self.n_cls), (1, 1), (1, 1), padding_mode='zeros', bias=False)
#
#     def forward(self, input_tensor):
#         # unit 1
#         x = self.conv_1(input_tensor)
#         x = self.bn_1(x)
#         x = self.lk_relu(x)
#         x = self.pool(x)
#
#         # unit 2
#         x = self.conv_2(x)
#         x = self.bn_2(x)
#         x = self.lk_relu(x)
#
#         # unit 3
#         x = self.conv_3(x)
#         x = self.bn_3(x)
#         x = self.lk_relu(x)
#         x = self.pool(x)
#
#         # unit 4
#         x = self.conv_4(x)
#         x = self.bn_4(x)
#         x = self.lk_relu(x)
#
#         # For skip connection
#         skip_x = x  # [b,64,80,80]
#         x = self.pool(x)
#
#         # unit 5
#         x = self.conv_5(x)
#         x = self.bn_5(x)
#         x = self.lk_relu(x)
#
#         # unit 6
#         x = self.conv_6(x)
#         x = self.bn_6(x)
#         x = self.lk_relu(x)
#
#         # unit 7
#         skip_x = self.conv_7(skip_x)
#         skip_x = self.bn_7(skip_x)
#         skip_x = self.lk_relu(skip_x)
#         skip_x = SpaceToDepth(block_size=2)(skip_x)
#
#         # concat
#         # [b,16,16,1024], [b,16,16,256],=> [b,16,16,1280]
#         x = torch.cat([skip_x, x], 1)
#
#         # unit 8
#         x = self.conv_8(x)
#         x = self.bn_8(x)
#         x = self.lk_relu(x)
#
#         # unit 9
#         x = self.conv_9(x)
#         x = x.reshape(-1, self.n_box, self.n_cls + 5, 40, 40)
#         return x.permute(0, 3, 4, 1, 2)
#
#
# class YoloV2Model(nn.Module):
#
#     def __init__(self, n_box=4, n_cls=4):
#         super(YoloV2Model, self).__init__()
#         self.gridsz = GRIDSZ
#         self.n_box = n_box
#         self.n_cls = n_cls
#
#         # layers define
#         self.lk_relu = nn.LeakyReLU(0.1)
#         self.pool = nn.MaxPool2d((2, 2))
#
#         # unit 1  (b, 1, 320, 320) -> (b, 16, 160, 160)
#         self.conv_1 = nn.Conv2d(1, 16, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_1 = nn.BatchNorm2d(16)
#
#         # unit 2  (b, 16, 160, 160) -> (b, 32, 160, 160)
#         self.conv_2 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_2 = nn.BatchNorm2d(32)
#
#         # unit 3  (b, 32, 160, 160) -> (b, 16, 160, 160)
#         self.conv_3 = nn.Conv2d(32, 16, (1, 1), stride=(1, 1), padding_mode='zeros', bias=False)
#         self.bn_3 = nn.BatchNorm2d(16)
#
#         # unit 4  (b, 16, 160, 160) -> (b, 32, 160, 160)
#         self.conv_4 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_4 = nn.BatchNorm2d(32)
#
#         # unit 5  (b, 32, 160, 160) -> (b, 16, 160, 160)
#         self.conv_5 = nn.Conv2d(32, 16, (1, 1), stride=(1, 1), padding_mode='zeros', bias=False)
#         self.bn_5 = nn.BatchNorm2d(16)
#
#         # unit 6  (b, 16, 160, 160) -> (b, 32, 160, 160)
#         self.conv_6 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_6 = nn.BatchNorm2d(32)
#
#         # unit 7  (b, 32, 160, 160) -> (b, 64, 80, 80)
#         self.conv_7 = nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_7 = nn.BatchNorm2d(64)
#
#         # unit 8  (b, 64, 80, 80) -> (b, 128, 80, 80)
#         self.conv_8 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_8 = nn.BatchNorm2d(128)
#
#         # unit 9  (b, 128, 80, 80) -> (b, 64, 80, 80)
#         self.conv_9 = nn.Conv2d(128, 64, (1, 1), stride=(1, 1), padding_mode='zeros', bias=False)
#         self.bn_9 = nn.BatchNorm2d(64)
#
#         # unit 10  (b, 64, 80, 80) -> (b, 128, 80, 80)
#         self.conv_10 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_10 = nn.BatchNorm2d(128)
#
#         # unit 11  (b, 128, 80, 80) -> (b, 64, 80, 80)
#         self.conv_11 = nn.Conv2d(128, 64, (1, 1), stride=(1, 1), padding_mode='zeros', bias=False)
#         self.bn_11 = nn.BatchNorm2d(64)
#
#         # unit 12  (b, 64, 80, 80) -> (b, 64, 80, 80)
#         self.conv_12 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_12 = nn.BatchNorm2d(64)
#
#         # unit 13  max_pooling -> (b, 40, 40, 128)
#         self.conv_13 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_13 = nn.BatchNorm2d(128)
#
#         # unit 14  (b, 128, 40, 40) -> (b, 256, 40, 40)
#         self.conv_14 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_14 = nn.BatchNorm2d(256)
#
#         # unit 15  (b, 256, 40, 40) -> (b, 128, 40, 40)
#         self.conv_15 = nn.Conv2d(256, 128, (1, 1), stride=(1, 1), padding_mode='zeros', bias=False)
#         self.bn_15 = nn.BatchNorm2d(128)
#
#         # unit 16  (b, 128, 40, 40) -> (b, 256, 40, 40)
#         self.conv_16 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_16 = nn.BatchNorm2d(256)
#
#         # unit 17  (b, 256, 40, 40) -> (b, 128, 40, 40)
#         self.conv_17 = nn.Conv2d(256, 128, (1, 1), stride=(1, 1), padding_mode='zeros', bias=False)
#         self.bn_17 = nn.BatchNorm2d(128)
#
#         # unit 18  (b, 128, 40, 40) -> (b, 256, 40, 40)
#         self.conv_18 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_18 = nn.BatchNorm2d(256)
#
#         # unit 19  (b, 256, 40, 40) -> (b, 256, 40, 40)
#         self.conv_19 = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_19 = nn.BatchNorm2d(256)
#
#         # unit 20  Skip_x block (b, 64, 80, 80) -> (b, 80, 80, 16)
#         self.conv_20 = nn.Conv2d(64, 16, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_20 = nn.BatchNorm2d(16)
#
#         # unit 21
#         self.conv_21 = nn.Conv2d(320, 256, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_21 = nn.BatchNorm2d(256)
#
#         # unit 22
#         self.conv_22 = nn.Conv2d(256, self.n_box*(5+self.n_cls), (1, 1), (1, 1), padding_mode='zeros', bias=False)
#
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#     def forward(self, input_tensor):
#         # unit 1
#         x = self.conv_1(input_tensor)
#         x = self.bn_1(x)
#         x = self.lk_relu(x)
#         x = self.pool(x)
#
#         # unit 2
#         x = self.conv_2(x)
#         x = self.bn_2(x)
#         x = self.lk_relu(x)
#
#         # unit 3
#         x = self.conv_3(x)
#         x = self.bn_3(x)
#         x = self.lk_relu(x)
#
#         # unit 4
#         x = self.conv_4(x)
#         x = self.bn_4(x)
#         x = self.lk_relu(x)
#
#         # unit 5
#         x = self.conv_5(x)
#         x = self.bn_5(x)
#         x = self.lk_relu(x)
#
#         # unit 6
#         x = self.conv_6(x)
#         x = self.bn_6(x)
#         x = self.lk_relu(x)
#
#         # unit 7
#         x = self.conv_7(x)
#         x = self.bn_7(x)
#         x = self.lk_relu(x)
#         x = self.pool(x)
#
#         # unit 8
#         x = self.conv_8(x)
#         x = self.bn_8(x)
#         x = self.lk_relu(x)
#
#         # unit 9
#         x = self.conv_9(x)
#         x = self.bn_9(x)
#         x = self.lk_relu(x)
#
#         # unit 10
#         x = self.conv_10(x)
#         x = self.bn_10(x)
#         x = self.lk_relu(x)
#
#         # unit 11
#         x = self.conv_11(x)
#         x = self.bn_11(x)
#         x = self.lk_relu(x)
#
#         # unit 12
#         x = self.conv_12(x)
#         x = self.bn_12(x)
#         x = self.lk_relu(x)
#
#         # For skip connection
#         skip_x = x  # [b,64,80,80]
#         x = self.pool(x)
#
#         # unit 13
#         x = self.conv_13(x)
#         x = self.bn_13(x)
#         x = self.lk_relu(x)
#
#         # unit 14
#         x = self.conv_14(x)
#         x = self.bn_14(x)
#         x = self.lk_relu(x)
#
#         # unit 15
#         x = self.conv_15(x)
#         x = self.bn_15(x)
#         x = self.lk_relu(x)
#
#         # unit 16
#         x = self.conv_16(x)
#         x = self.bn_16(x)
#         x = self.lk_relu(x)
#
#         # unit 17
#         x = self.conv_17(x)
#         x = self.bn_17(x)
#         x = self.lk_relu(x)
#
#         # unit 18
#         x = self.conv_18(x)
#         x = self.bn_18(x)
#         x = self.lk_relu(x)
#
#         # unit 19
#         x = self.conv_19(x)
#         x = self.bn_19(x)
#         x = self.lk_relu(x)
#
#         # unit 21
#         skip_x = self.conv_20(skip_x)
#         skip_x = self.bn_20(skip_x)
#         skip_x = self.lk_relu(skip_x)
#         skip_x = SpaceToDepth(block_size=2)(skip_x)
#
#         # concat
#         # [b,16,16,1024], [b,16,16,256],=> [b,16,16,1280]
#         x = torch.cat([skip_x, x], 1)
#
#         # unit 22
#         x = self.conv_21(x)
#         x = self.bn_21(x)
#         x = self.lk_relu(x)
#         x = nn.Dropout(0.5)(x)
#
#         # unit 23
#         x = self.conv_22(x)
#         x = x.reshape(-1, self.n_box, self.n_cls + 5, 40, 40)
#         return x.permute(0, 3, 4, 1, 2)
#
#
# class YoloV4Model(nn.Module):
#     def __init__(self, n_box=4, n_cls=4):
#         super(YoloV4Model, self).__init__()
#         self.gridsz = GRIDSZ
#         self.n_box = n_box
#         self.n_cls = n_cls
#
#         # layers define
#         self.lk_relu = nn.LeakyReLU(0.1)
#         self.pool = nn.MaxPool2d((2, 2))
#
#         # unit 1  (b, 1, 320, 320) -> (b, 8, 160, 160)
#         self.conv_1 = nn.Conv2d(1, 8, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_1 = nn.BatchNorm2d(8)
#
#         # unit 2  (b, 8, 160, 160) -> (b, 16, 160, 160)
#         self.conv_2 = nn.Conv2d(8, 16, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_2 = nn.BatchNorm2d(16)
#
#         # unit 3  (b, 16, 160, 160) -> (b, 32, 80, 80)
#         self.conv_3 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_3 = nn.BatchNorm2d(32)
#
#         # unit 4  (b, 32, 80, 80) -> (b, 64, 80, 80)
#         self.conv_4 = nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_4 = nn.BatchNorm2d(64)
#
#         # unit 5  max_pooling -> (b, 40, 40, 128)
#         self.conv_5 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_5 = nn.BatchNorm2d(128)
#
#         # unit 6  (b, 128, 40, 40) -> (b, 256, 40, 40)
#         self.conv_6 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
#         self.bn_6 = nn.BatchNorm2d(256)
#
#         # unit 7  (b, 256, 40, 40) -> (b, 4*9, 40, 40)
#         self.conv_7 = nn.Conv2d(256, self.n_box*(5+self.n_cls), (1, 1), stride=(1, 1), padding_mode='zeros', bias=False)
#
#     def forward(self, input_tensor):
#         # unit 1
#         x = self.conv_1(input_tensor)
#         x = self.bn_1(x)
#         x = self.lk_relu(x)
#         x = self.pool(x)
#
#         # unit 2
#         x = self.conv_2(x)
#         x = self.bn_2(x)
#         x = self.lk_relu(x)
#
#         # unit 3
#         x = self.conv_3(x)
#         x = self.bn_3(x)
#         x = self.lk_relu(x)
#         x = self.pool(x)
#
#         # unit 4
#         x = self.conv_4(x)
#         x = self.bn_4(x)
#         x = self.lk_relu(x)
#
#         # unit 5
#         x = self.conv_5(x)
#         x = self.bn_5(x)
#         x = self.lk_relu(x)
#         x = self.pool(x)
#
#         # unit 6
#         x = self.conv_6(x)
#         x = self.bn_6(x)
#         x = self.lk_relu(x)
#
#         # unit 9
#         x = self.conv_7(x)
#         x = x.reshape(-1, self.n_box, self.n_cls + 5, 40, 40)
#         return x.permute(0, 3, 4, 1, 2)


class YoloV5Model(nn.Module):
    def __init__(self, n_box=4, n_cls=4, attention_layer=6):
        super(YoloV5Model, self).__init__()

        self.gridsz = GRIDSZ
        self.n_box = n_box
        self.n_cls = n_cls
        self.attention_layer = attention_layer

        # layers define
        self.lk_relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d((2, 2))

        # unit 1  (b, 1, 320, 320) -> (b, 8, 160, 160)
        self.conv_1 = nn.Conv2d(1, 8, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
        self.bn_1 = nn.BatchNorm2d(8)

        # unit 2  (b, 8, 160, 160) -> (b, 16, 160, 160)
        self.conv_2 = nn.Conv2d(8, 16, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
        self.bn_2 = nn.BatchNorm2d(16)

        # unit 3  (b, 16, 160, 160) -> (b, 32, 80, 80)
        self.conv_3 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
        self.bn_3 = nn.BatchNorm2d(32)

        # unit 4  (b, 32, 80, 80) -> (b, 64, 80, 80)
        self.conv_4 = nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
        self.bn_4 = nn.BatchNorm2d(64)

        # unit 5  max_pooling -> (b, 40, 40, 128)
        self.conv_5 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
        self.bn_5 = nn.BatchNorm2d(128)

        # unit 6  (b, 128, 40, 40) -> (b, 256, 40, 40)
        self.conv_6 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
        self.bn_6 = nn.BatchNorm2d(256)

        # unit 7  (b, 256, 40, 40) -> (b, 4*9, 40, 40)
        self.conv_7 = nn.Conv2d(256, self.n_box*(5+self.n_cls), (1, 1), stride=(1, 1),
                                padding_mode='zeros', bias=False)

        # Channel and Spatial attention
        # The 'in_feature' param of CAB in instance generating is decided according to
        # input feature channel size, if cab and sab used after conv_6, it will be 256,
        # otherwise it is  self.n_box*(5+self.n_cls) (used after conv_7).
        self.cab = CAB(256) if attention_layer == 6 else CAB(self.n_box*(5+self.n_cls))
        self.sab = SAB()

    def forward(self, input_tensor):
        # unit 1
        x = self.conv_1(input_tensor)
        x = self.bn_1(x)
        x = self.lk_relu(x)
        x = self.pool(x)

        # unit 2
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.lk_relu(x)

        # unit 3
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.lk_relu(x)
        x = self.pool(x)

        # unit 4
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.lk_relu(x)

        # unit 5
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.lk_relu(x)
        x = self.pool(x)

        # unit 6
        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.lk_relu(x)
        if self.attention_layer == 6:
            x = self.cab(x) * x
            x = self.sab(x) * x

        # unit 9
        x = self.conv_7(x)
        if self.attention_layer == 7:
            x = self.cab(x) * x
            x = self.sab(x) * x

        x = x.reshape(-1, self.n_box, self.n_cls + 5, 40, 40)
        return x.permute(0, 3, 4, 1, 2)


if __name__ == "__main__":
    input_tensor = torch.randn(1, 1, 320, 320)
    model = YoloV5Model(attention_layer=7)
    output = model(input_tensor)
    print(output.shape)
