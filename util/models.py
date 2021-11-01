import torch
import torch.nn as nn


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
        pool_out = self.fc2(self.relu(self.fc1(self.maxpooling(x))))
        return self.sigmoid(pool_out)


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
    def __init__(self, filters_in, filters_out, kernel_size=(3, 3), pooling=False, **kwargs):
        super(ConvBlock, self).__init__()
        # layers define
        conv2d_params = {"stride": (1, 1), "padding": 1, "padding_mode": 'zeros', "bias": False}
        conv2d_params.update({k: v for k, v in kwargs.items() if k in conv2d_params})
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, **conv2d_params)
        self.bn = nn.BatchNorm2d(filters_out)
        self.elu = nn.ELU()
        if pooling:
            self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        return x


class YoloV5Model(nn.Module):
    def __init__(self, n_box=4, n_cls=3, grid_size=40, attention_layer=6):
        super(YoloV5Model, self).__init__()
        self.gridsz  =   grid_size
        self.n_box   =   n_box
        self.n_cls   =   n_cls
        self.attention_layer = attention_layer

        # layers define
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d((2, 2))

        conv2d_params = {"stride": (1, 1), "padding": 1, "padding_mode": 'zeros', "bias": False}
        # unit 1  (b, 4, 320, 320) -> (b, 8, 160, 160)
        self.conv_1 = nn.Conv2d(4, 8, (3, 3), **conv2d_params)
        self.bn_1 = nn.BatchNorm2d(8)

        # unit 2  (b, 8, 160, 160) -> (b, 16, 160, 160)
        self.conv_2 = nn.Conv2d(8, 16, (3, 3), **conv2d_params)
        self.bn_2 = nn.BatchNorm2d(16)

        # unit 3  (b, 16, 160, 160) -> (b, 32, 80, 80)
        self.conv_3 = nn.Conv2d(16, 32, (3, 3), **conv2d_params)
        self.bn_3 = nn.BatchNorm2d(32)

        # unit 4  (b, 32, 80, 80) -> (b, 64, 80, 80)
        self.conv_4 = nn.Conv2d(32, 64, (3, 3), **conv2d_params)
        self.bn_4 = nn.BatchNorm2d(64)

        # unit 5  max_pooling -> (b, 40, 40, 128)
        self.conv_5 = nn.Conv2d(64, 128, (3, 3), **conv2d_params)
        self.bn_5 = nn.BatchNorm2d(128)

        # unit 6  (b, 128, 40, 40) -> (b, 256, 40, 40)
        self.conv_6 = nn.Conv2d(128, 256, (3, 3), **conv2d_params)
        self.bn_6 = nn.BatchNorm2d(256)

        # unit 7  (b, 256, 40, 40) -> (b, 4*9, 40, 40)
        self.conv_7 = nn.Conv2d(256, self.n_box*(5+self.n_cls), **conv2d_params)

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
        x = self.elu(x)
        x = self.pool(x)

        # unit 2
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.elu(x)

        # unit 3
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.elu(x)
        x = self.pool(x)

        # unit 4
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.elu(x)

        # unit 5
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.elu(x)
        x = self.pool(x)

        # unit 6
        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.elu(x)
        if self.attention_layer == 6:
            x = self.cab(x) * x
            x = self.sab(x) * x

        # unit 9
        x = self.conv_7(x)
        if self.attention_layer == 7:
            x = self.cab(x) * x
            x = self.sab(x) * x

        x = x.reshape(-1, self.n_box, self.n_cls+5, self.gridsz, self.gridsz)
        return x.permute(0, 3, 4, 1, 2)


class YoloV6Model(nn.Module):
    def __init__(self, n_box=4, n_cls=3, grid_size=40):
        super(YoloV6Model, self).__init__()
        self.gridsz  =   grid_size
        self.n_box   =   n_box
        self.n_cls   =   n_cls

        # layers define
        conv2d_params = {"stride": (1, 1), "padding": 1, "padding_mode": 'zeros', "bias": False}
        blocks_parameters = [(4, 8, (3, 3), True),
                             (8, 16, (3, 3), False),
                             (16, 32, (3, 3), True),
                             (32, 64, (3, 3), False),
                             (64, 128, (3, 3), True),
                             (128, 256, (3, 3), False)]

        self.conv_blocks = torch.nn.Sequential(
            *[ConvBlock(*block_param, **conv2d_params) for block_param in blocks_parameters],
            nn.Conv2d(256, self.n_box*(5+self.n_cls), (3, 3), **conv2d_params)
        )
        # unit 1  (b, 4, 320, 320) -> (b, 8, 160, 160)
        # unit 2  (b, 8, 160, 160) -> (b, 16, 160, 160)
        # unit 3  (b, 16, 160, 160) -> (b, 32, 80, 80)
        # unit 4  (b, 32, 80, 80) -> (b, 64, 80, 80)
        # unit 5  max_pooling -> (b, 40, 40, 128)
        # unit 6  (b, 128, 40, 40) -> (b, 256, 40, 40)
        # unit 7  (b, 256, 40, 40) -> (b, 4*9, 40, 40)

        # Channel and Spatial attention
        # The 'in_feature' param of CAB in instance generating is decided according to
        # input feature channel size, if cab and sab used after conv_6, it will be 256,
        # otherwise it is  self.n_box*(5+self.n_cls) (used after conv_7).
        self.cab = CAB(self.n_box*(5+self.n_cls))
        self.sab = SAB()

    def forward(self, x):
        x = self.conv_blocks(x)
        # unit 9
        x = self.cab(x) * x
        x = self.sab(x) * x
        x = x.reshape(-1, self.n_box, self.n_cls+5, self.gridsz, self.gridsz)
        return x.permute(0, 3, 4, 1, 2)
