from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1, drop_rate=0.1):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(drop_rate),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.Dropout(drop_rate),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class IA(nn.Module):
    def __init__(self, channel, ):
        super().__init__()
        self.r1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False, dilation=1)
        self.r2 = nn.Conv2d(channel, channel, kernel_size=3, padding=3, bias=False, dilation=3)

        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        r1 = self.relu(self.bn(self.r1(x)))
        r2 = self.relu(self.bn(self.r2(x)))
        r1_sig = self.sig(r1)
        r2_sig = self.sig(r2)
        r1_o = (1 - r1_sig) * r2_sig + r1
        r2_o = (1 - r2_sig) * r1_sig + r2
        output = r1_o + r2_o + x

        return output


class IA2(nn.Module):
    def __init__(self, channel, ):
        super().__init__()
        self.r1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False, dilation=1)
        self.r2 = nn.Conv2d(channel, channel, kernel_size=3, padding=3, bias=False, dilation=3)
        self.out = DoubleConv(channel * 2, channel, channel)

        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        r1 = self.relu(self.bn(self.r1(x)))
        r2 = self.relu(self.bn(self.r2(x)))
        r1_sig = self.sig(r1)
        r2_sig = self.sig(r2)
        r1_o = (1 - r1_sig) * r2_sig + r1
        r2_o = (1 - r2_sig) * r1_sig + r2
        output = torch.cat((r1_o, r2_o), dim=1)
        output = self.out(output)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, channel, ):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class PBlock2(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super().__init__()
        self.sa1 = SpatialAttention(kernel_size=kernel_size)
        self.sa2 = SpatialAttention(kernel_size=kernel_size)
        self.ca1 = ChannelAttention(channel)
        self.ca2 = ChannelAttention(channel)
        self.r1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False, dilation=1)
        self.r2 = nn.Conv2d(channel, channel, kernel_size=3, padding=3, bias=False, dilation=3)
        self.r3 = nn.Conv2d(channel, channel, kernel_size=3, padding=5, bias=False, dilation=5)
        self.c1 = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, bias=False)
        self.c2 = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, bias=False)
        self.conv = nn.Conv2d(channel, channel, kernel_size=1)

        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r1 = self.relu(self.bn(self.r1(x)))
        r2 = self.relu(self.bn(self.r2(x)))
        r3 = self.relu(self.bn(self.r3(x)))
        c1 = self.relu(self.bn(self.c1(torch.cat((r1, r2), dim=1))))
        c2 = self.relu(self.bn(self.c2(torch.cat((r2, r3), dim=1))))
        c1_sa = self.sa1(c1)
        c2_sa = self.sa2(c2)
        c1_ca = self.ca1(c1)
        c2_ca = self.ca2(c2)

        c1_att = c1_sa * c2_ca
        c2_att = c1_ca * c2_sa
        att = c1_att + c2_att
        out = x * att + x
        out = self.relu(self.bn(self.conv(out)))

        return out


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, drop_rate=0.1):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels, drop_rate=drop_rate)
        )


class EdgeUp(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.1):
        super(EdgeUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, drop_rate=drop_rate)

        self.sig = nn.Sigmoid()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x1_sig = 1 - self.sig(x1)
        x2_sig = 1 - self.sig(x2)
        x1_edge = x1_sig * self.sig(x2)
        x2_edge = x2_sig * self.sig(x1)
        x1 = x1_edge + x1
        x2 = x2_edge + x2

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class EdgeUp2(nn.Module):
    def __init__(self, in_channels, out_channels, upon=True):
        super(EdgeUp2, self).__init__()
        if upon:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        self.sig = nn.Sigmoid()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        skip = x2
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x1_sig = 1 - self.sig(x1)
        x2_sig = 1 - self.sig(x2)
        x1_edge = x1_sig * self.sig(x2)
        x2_edge = x2_sig * self.sig(x1)
        x1 = x1_edge + x1
        x2 = x2_edge + x2

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x) + skip
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Aux(nn.Module):
    def __init__(self, in_channels, uprate, drop_rate):
        super(Aux, self).__init__()
        self.up = nn.Upsample(scale_factor=uprate, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, 1, in_channels // 2, drop_rate=drop_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = self.conv(x1)

        return x


class CCV4(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 base_c: int = 64):
        super(CCV4, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.in_conv = DoubleConv(in_channels, base_c, drop_rate=0.1)
        self.down1 = Down(base_c, base_c * 2, drop_rate=0.1)
        self.down2 = Down(base_c * 2, base_c * 4, drop_rate=0.1)
        self.down3 = Down(base_c * 4, base_c * 8, drop_rate=0.1)
        self.down4 = Down(base_c * 8, base_c * 8, drop_rate=0.1)
        self.up1 = EdgeUp(base_c * 16, base_c * 4, drop_rate=0.1)
        self.up2 = EdgeUp(base_c * 8, base_c * 2, drop_rate=0.1)
        self.up3 = EdgeUp(base_c * 4, base_c, drop_rate=0.1)
        self.up4 = EdgeUp(base_c * 2, base_c, drop_rate=0.1)

        self.block1 = EdgeUp2(base_c * 2, base_c)
        self.block2 = EdgeUp2(base_c * 4, base_c * 2)
        self.block3 = EdgeUp2(base_c * 8, base_c * 4)
        self.block4 = EdgeUp2(base_c * 16, base_c * 8, upon=False)

        # self.block1 = IA2(base_c)
        # self.block2 = IA2(base_c * 2)
        # self.block3 = IA2(base_c * 4)
        # self.block4 = IA2(base_c * 8)
        self.block = PBlock2(base_c * 8)

        self.aux1 = nn.Sequential(
            nn.Conv2d(base_c * 4, base_c, kernel_size=1), nn.BatchNorm2d(base_c), nn.PReLU()
        )
        self.aux2 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c, kernel_size=1), nn.BatchNorm2d(base_c), nn.PReLU()
        )
        self.aux3 = nn.Sequential(
            nn.Conv2d(base_c, base_c, kernel_size=1), nn.BatchNorm2d(base_c), nn.PReLU()
        )
        self.aux4 = nn.Sequential(
            nn.Conv2d(base_c, base_c, kernel_size=1), nn.BatchNorm2d(base_c), nn.PReLU()
        )
        self.predict4 = nn.Conv2d(base_c, 1, kernel_size=1)
        self.predict3 = nn.Conv2d(base_c, 1, kernel_size=1)
        self.predict2 = nn.Conv2d(base_c, 1, kernel_size=1)
        self.predict1 = nn.Conv2d(base_c, 1, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(base_c * 4, base_c, kernel_size=3, padding=1), nn.BatchNorm2d(base_c), nn.PReLU(),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1), nn.BatchNorm2d(base_c), nn.PReLU(),
            nn.Conv2d(base_c, base_c, kernel_size=1), nn.BatchNorm2d(base_c), nn.PReLU()
        )
        self.attention4 = SpatialAttention()
        self.attention3 = SpatialAttention()
        self.attention2 = SpatialAttention()
        self.attention1 = SpatialAttention()
        self.refine4 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c, kernel_size=3, padding=1), nn.BatchNorm2d(base_c), nn.PReLU(),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1), nn.BatchNorm2d(base_c), nn.PReLU(),
            nn.Conv2d(base_c, base_c, kernel_size=1), nn.BatchNorm2d(base_c), nn.PReLU()
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c, kernel_size=3, padding=1), nn.BatchNorm2d(base_c), nn.PReLU(),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1), nn.BatchNorm2d(base_c), nn.PReLU(),
            nn.Conv2d(base_c, base_c, kernel_size=1), nn.BatchNorm2d(base_c), nn.PReLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c, kernel_size=3, padding=1), nn.BatchNorm2d(base_c), nn.PReLU(),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1), nn.BatchNorm2d(base_c), nn.PReLU(),
            nn.Conv2d(base_c, base_c, kernel_size=1), nn.BatchNorm2d(base_c), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c, kernel_size=3, padding=1), nn.BatchNorm2d(base_c), nn.PReLU(),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1), nn.BatchNorm2d(base_c), nn.PReLU(),
            nn.Conv2d(base_c, base_c, kernel_size=1), nn.BatchNorm2d(base_c), nn.PReLU()
        )
        self.predict4_2 = nn.Conv2d(base_c, 1, kernel_size=1)
        self.predict3_2 = nn.Conv2d(base_c, 1, kernel_size=1)
        self.predict2_2 = nn.Conv2d(base_c, 1, kernel_size=1)
        self.predict1_2 = nn.Conv2d(base_c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        input = x
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1 = self.block1(x2, x1)
        x2 = self.block2(x3, x2)
        x3 = self.block3(x4, x3)
        x4 = self.block4(x5, x4)
        x5 = self.block(x5)

        x_aux1 = self.up1(x5, x4)
        x_aux2 = self.up2(x_aux1, x3)
        x_aux3 = self.up3(x_aux2, x2)
        x_aux4 = self.up4(x_aux3, x1)

        x_aux1 = self.aux1(x_aux1)
        x_aux2 = self.aux2(x_aux2)
        x_aux3 = self.aux3(x_aux3)
        x_aux4 = self.aux4(x_aux4)

        x_aux1 = F.upsample(x_aux1, size=input.size()[2:], mode='bilinear')
        x_aux2 = F.upsample(x_aux2, size=input.size()[2:], mode='bilinear')
        x_aux3 = F.upsample(x_aux3, size=input.size()[2:], mode='bilinear')

        predict1 = self.predict1(x_aux1)
        predict2 = self.predict2(x_aux2)
        predict3 = self.predict3(x_aux3)
        predict4 = self.predict1(x_aux4)

        fuse = self.fuse(torch.cat((x_aux4, x_aux3, x_aux2, x_aux1), 1))
        attention4 = self.attention4(x_aux4 + fuse)
        attention3 = self.attention3(x_aux3 + fuse)
        attention2 = self.attention2(x_aux2 + fuse)
        attention1 = self.attention1(x_aux1 + fuse)
        refine4 = self.refine4(torch.cat((x_aux4, attention4 * fuse), 1))
        refine3 = self.refine3(torch.cat((x_aux3, attention3 * fuse), 1))
        refine2 = self.refine2(torch.cat((x_aux2, attention2 * fuse), 1))
        refine1 = self.refine1(torch.cat((x_aux1, attention1 * fuse), 1))
        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict1_2 = self.predict1_2(refine1)

        final = predict1_2 + predict2_2 + predict3_2 + predict4_2

        result = OrderedDict()
        result["final"] = final
        result["predict1"] = predict1
        result["predict2"] = predict2
        result["predict3"] = predict3
        result["predict4"] = predict4
        result["predict1_2"] = predict1_2
        result["predict2_2"] = predict2_2
        result["predict3_2"] = predict3_2
        result["predict4_2"] = predict4_2

        return result


if __name__ == '__main__':
    net = CCV4()

    pred = Variable(torch.randn(1, 1, 240, 240))
    out = net(pred)
    print(out["final"].size())
