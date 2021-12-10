import torch
import torch.nn as nn


class CAB(nn.Module):
    # channel attention
    def __init__(self, in_features):
        super(CAB, self).__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)   # (b, C, H, W) -> (b, C, 1)
        self.fc1 = nn.Conv2d(in_features, in_features // 4, 1, bias=False)  # (b, C, 1) -> (b, C/4, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_features // 4, in_features, 1, bias=False)  # (b, C/4, 1) -> (b, C, 1)
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


class YoloV6Model(nn.Module):
    def __init__(self, input_channel=4, n_box=4, n_cls=3, grid_size=40):
        super(YoloV6Model, self).__init__()
        self.gridsz  =   grid_size
        self.n_box   =   n_box
        self.n_cls   =   n_cls

        # layers define
        conv2d_params = {"stride": (1, 1), "padding": 1, "padding_mode": 'zeros', "bias": False}
        blocks_parameters = [(input_channel, 8, (3, 3), True),  # unit 1  (b, 4, 320, 320)  ->  (b, 8, 160, 160)
                             (8, 16, (3, 3), False),            # unit 2  (b, 8, 160, 160)  ->  (b, 16, 160, 160
                             (16, 32, (3, 3), True),            # unit 3  (b, 16, 160, 160) ->  (b, 32, 80, 80)
                             (32, 64, (3, 3), False),           # unit 4  (b, 32, 80, 80)   ->  (b, 64, 80, 80)
                             (64, 128, (3, 3), True),           # unit 5  (b, 64, 80, 80)   ->  (b, 128, 40, 40)
                             (128, 256, (3, 3), False)]         # unit 6  (b, 128, 40, 40)  ->  (b, 256, 40, 40)

        self.conv_blocks = torch.nn.Sequential(
            # unit 1~6 ()
            *[ConvBlock(*block_param, **conv2d_params) for block_param in blocks_parameters],
            # unit 7 (Fusion block)  (b, 256, 40, 40)  ->  (b, 4*9, 40, 40)
            nn.Conv2d(256, self.n_box*(5+self.n_cls), (1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_box*(5+self.n_cls)),
            nn.ELU()
        )
        # Channel and Spatial attention
        # The 'in_feature' param of CAB in instance generating is decided according to
        # input feature channel size, which here is  self.n_box*(5+self.n_cls) (used after conv_7).
        self.cab = CAB(self.n_box*(5+self.n_cls))
        self.sab = SAB()

    def forward(self, x):
        x = self.conv_blocks(x)
        # unit 9
        x = self.cab(x) * x
        x = self.sab(x) * x
        x = x.reshape(-1, self.n_box, self.n_cls+5, self.gridsz, self.gridsz)
        return x.permute(0, 3, 4, 1, 2)
