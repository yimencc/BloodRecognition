import torch
import torch.nn as nn

from Deeplearning.dataset import GRIDSZ


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

        # unit 1  (b, 4, 320, 320) -> (b, 8, 160, 160)
        self.conv_1 = nn.Conv2d(4, 8, (3, 3), stride=(1, 1), padding=1, padding_mode='zeros', bias=False)
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
    input_tensor = torch.randn(1, 4, 320, 320)
    model = YoloV5Model(attention_layer=7)
    output = model(input_tensor)
    print(output.shape)
