import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101

def convbatchrelu(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )


def batchconv(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1)
        for t in range(4):
            if t == 0:
                self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))

    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t == 0:
                self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))

    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class downsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase) - 1):
            if residual_on:
                self.down.add_module('res_down_%d' % n, resdown(nbase[n], nbase[n + 1], sz))
            else:
                self.down.add_module('conv_down_%d' % n, convdown(nbase[n], nbase[n + 1], sz))

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.concatenation = concatenation
        if concatenation:
            self.conv = batchconv(in_channels * 2, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels * 2)
        else:
            self.conv = batchconv(in_channels, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            if self.concatenation:
                x = torch.cat((y, x), dim=1)
            else:
                x = x + y
        feat = self.full(style)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat.unsqueeze(-1).unsqueeze(-1)).to_mkldnn()
        else:
            y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y


class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz,
                                                      concatenation=concatenation))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.proj = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x


class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz,
                                                      concatenation=concatenation))

    def forward(self, x, y, style, mkldnn=False):
        x = self.conv[1](style, self.conv[0](x), y=y)
        return x


class make_style(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        # style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, axis=1, keepdim=True) ** .5

        return style


class upsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, concatenation=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            if residual_on:
                self.up.add_module('res_up_%d' % (n - 1),
                                   resup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation))
            else:
                self.up.add_module('conv_up_%d' % (n - 1),
                                   convup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation))

    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up) - 2, -1, -1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x


class CPNet(nn.Module):
    def __init__(self, n_channels, n_classes, sz=3,
                 residual_on=True, style_on=True,
                 concatenation=False, mkldnn=False,
                 diam_mean=30.):
        super(CPNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.nbase = [n_channels, 32, 64, 128, 256]
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(self.nbase, sz, residual_on=residual_on)
        nbaseup = self.nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], n_classes, 1)
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.style_on = style_on

    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T0 = self.upsample(style, T0, self.mkldnn)
        T0 = self.output(T0)
        if self.mkldnn:
            T0 = T0.to_dense()
        return T0, style0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            state_dict = torch.load(filename)
        else:
            self.__init__(self.n_channels,
                          self.n_classes,
                          self.sz,
                          self.residual_on,
                          self.style_on,
                          self.concatenation,
                          self.mkldnn,
                          self.diam_mean)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)

################################################################################################################################
################################################################################################################################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, x5

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            state_dict = torch.load(filename)
        else:
            self.__init__(self.n_channels, self.n_classes, bilinear=self.bilinear)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)

################################################################################################################################
################################################################################################################################


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithDResNet(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x1 = self.upsample(up_x)
        diffY = down_x.size()[2] - x1.size()[2]
        diffX = down_x.size()[3] - x1.size()[3]
        x = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class DResUNet(nn.Module):
    DEPTH = 6
    def __init__(self, n_channels, n_classes, backbone="resnet50"):
        super(DResUNet, self).__init__()
        if backbone == "resnet50":
            resnet = resnet50(pretrained=True)
        elif backbone == "resnet101":
            resnet = resnet101(pretrained=True)
        else:
            raise NotImplementedError(f"Backbone {backbone} is not implemented yet.")
        self.backbone = backbone
        self.n_channels = n_channels
        self.n_classes = n_classes
        down_blocks = []
        up_blocks = []
        # self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_block = nn.Sequential(*(list([nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)]) + list(resnet.children())[1:3]))
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithDResNet(2048, 1024))
        up_blocks.append(UpBlockForUNetWithDResNet(1024, 512))
        up_blocks.append(UpBlockForUNetWithDResNet(512, 256))
        up_blocks.append(UpBlockForUNetWithDResNet(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithDResNet(in_channels=64 + n_channels, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=True):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)
        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (DResUNet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x
        x = self.bridge(x)
        output_feature_map = x
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{DResUNet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        # output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x


################################################################################################################################
################################################################################################################################
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        aa = torch.unsqueeze(torch.mean(x, dim=1), 1)
        ag = torch.unsqueeze(torch.max(x, dim=1)[0], 1)
        a = torch.cat([ag, aa], dim=1)
        a = self.sigmoid(self.conv(a))
        x = x*a
        return x

class CPNetSAM(nn.Module):
    def __init__(self, n_channels, n_classes, sz=3,
                 residual_on=True, style_on=True,
                 concatenation=False, mkldnn=False,
                 diam_mean=30., sam_position="mid"):

        super(CPNetSAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.nbase = [n_channels, 32, 64, 128, 256]
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(self.nbase, sz, residual_on=residual_on)
        nbaseup = self.nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], self.n_classes, 1)
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.style_on = style_on
        self.sam_position = sam_position
        self.sam = SpatialAttentionModule()

    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        if self.sam_position == "mid":
            T0[-1] = self.sam(T0[-1])
        T0 = self.upsample(style, T0, self.mkldnn)
        if self.sam_position == "tail":
            T0 = self.sam(T0)
        T0 = self.output(T0)
        if self.mkldnn:
            T0 = T0.to_dense()
        return T0, style0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            state_dict = torch.load(filename)
        else:
            self.__init__(self.n_channels,
                          self.n_classes,
                          self.sz,
                          self.residual_on,
                          self.style_on,
                          self.concatenation,
                          self.mkldnn,
                          self.diam_mean,
                          self.sam_position)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)


class UNetSAM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, sam_position="mid"):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sam_position = sam_position
        self.sam = SpatialAttentionModule()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.sam_position == "mid":
            x5 = self.sam(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.sam_position == "tail":
            x = self.sam(x)
        logits = self.outc(x)
        return logits, x5

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            state_dict = torch.load(filename)
        else:
            self.__init__(self.n_channels, self.n_classes, bilinear=self.bilinear, sam_position=self.sam_position)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)


class DResUNetSAM(nn.Module):
    DEPTH = 6
    def __init__(self, n_channels, n_classes, backbone="resnet50", sam_position="mid"):
        super(DResUNetSAM, self).__init__()
        if backbone == "resnet50":
            resnet = resnet50(pretrained=True)
        elif backbone == "resnet101":
            resnet = resnet101(pretrained=True)
        else:
            raise NotImplementedError(f"Backbone {backbone} is not implemented yet.")
        self.backbone = backbone
        self.n_channels = n_channels
        self.n_classes = n_classes
        down_blocks = []
        up_blocks = []
        # self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_block = nn.Sequential(*(list([nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)]) + list(resnet.children())[1:3]))
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithDResNet(2048, 1024))
        up_blocks.append(UpBlockForUNetWithDResNet(1024, 512))
        up_blocks.append(UpBlockForUNetWithDResNet(512, 256))
        up_blocks.append(UpBlockForUNetWithDResNet(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithDResNet(in_channels=64 + n_channels, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        self.sam_position = sam_position
        self.sam = SpatialAttentionModule()

    def forward(self, x, with_output_feature_map=True):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)
        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (DResUNet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x
        x = self.bridge(x)
        output_feature_map = x
        if self.sam_position == "mid":
            x = self.sam(x)
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{DResUNet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        # output_feature_map = x
        if self.sam_position == "tail":
            x = self.sam(x)
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            state_dict = torch.load(filename)
        else:
            self.__init__(self.n_channels, self.n_classes, backbone=self.backbone, sam_position=self.sam_position)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)
