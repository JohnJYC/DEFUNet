""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    # """Downscaling with maxpool then double deform conv"""
    #
    # def __init__(self, in_channels, out_channels):
    #     super().__init__()
    #     self.maxpool = nn.MaxPool2d(2)
    #     self.offsets_conv1 = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
    #     self.conv1 = DeformConv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
    #     self.bn1 = nn.BatchNorm2d(out_channels // 2)
    #     self.relu1 = nn.ReLU(inplace=True)
    #     self.offsets_conv2 = nn.Conv2d(out_channels // 2, 18, kernel_size=3, padding=1)
    #     self.conv2 = DeformConv2d(out_channels // 2, out_channels, kernel_size=3, padding=1)
    #     self.bn2 = nn.BatchNorm2d(out_channels)
    #     self.relu2 = nn.ReLU(inplace=True)
    #
    # def forward(self, x):
    #     x = self.maxpool(x)
    #     offsets1 = self.offsets_conv1(x)
    #     x = self.conv1(x, offsets1)
    #     x = self.bn1(x)
    #     x = self.relu1(x)
    #
    #     offsets2 = self.offsets_conv2(x)
    #     x = self.conv2(x, offsets2)
    #     x = self.bn2(x)
    #     x = self.relu2(x)
    #     return x
    """Downscaling with maxpool, dropout, residual connection and DeformConv"""

    def __init__(self, in_channels, out_channels, dropout_ratio=0.2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout_ratio)

        # First deform conv branch
        self.offsets_conv1 = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        self.conv1 = DeformConv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)

        # Second deform conv branch (with residual connection)
        self.offsets_conv2 = nn.Conv2d(out_channels // 2, 18, kernel_size=3, padding=1)
        self.conv2 = DeformConv2d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = x  # Store input for residual connection

        x = self.maxpool(x)
        x = self.dropout(x)  # Add dropout for regularization

        offsets1 = self.offsets_conv1(x)
        x = self.conv1(x, offsets1)
        x = self.bn1(x)
        x = self.relu1(x)

        offsets2 = self.offsets_conv2(x)
        x = self.conv2(x, offsets2)
        x = self.bn2(x)

        # Add residual connection
        x = x + shortcut

        return x

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

from torch.autograd import Variable, Function
import torch
from torch import nn
import numpy as np



class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    # 注意，offset的Tensor尺寸是[b, 18, 3, 3]
    def forward(self, x, offset):

        dtype = offset.data.type()
        ks = self.kernel_size
        # N=9=3x3
        N = offset.size(1) // 2

        # 将offset的顺序从[x1, y1, x2, y2, ...] 改成[x1, x2..., y1, y2, ...]
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        # torch.unsqueeze()是为了增加维度
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        # 根据维度dim按照索引列表index从offset中选取指定元素
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        # 对输入x进行padding
        if self.padding:
            x = self.zero_padding(x)

        # 将offset放到网格上，也即是图中所示的那个绿色带箭头的offset图
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # 维度变换
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # floor是向下取整 q_lt（左上角采样点）
        q_lt = Variable(p.data, requires_grad=False).floor()
        # +1相当于向上取整
        q_rb = q_lt + 1

        # 将lt限制在图像范围内
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # 将rb限制在图像范围内
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # 获得lb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        # 获得rt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # 插值的时候需要考虑一下padding对原始索引的影响
        # (b, h, w, N)
        # torch.lt() 逐元素比较input和other，即是否input < other
        # torch.rt() 逐元素比较input和other，即是否input > other
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        # 相当于求双线性插值中的u和v
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # 插值的4个系数 g_lt（左上角采样点的权重）
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N) x_q_lt（pixel value）
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # 插值的最终操作在这里
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        # N = 9, h = 3, w = 3
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        return x_offset