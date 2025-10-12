
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Conv2d
from module.ResBlock import ResBlock

class GSA(nn.Module):
    def __init__(self, channels, num_heads=8, bias=False):
        super(GSA, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        if prev_atns is None:
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = self.act(attn)
            out = (attn @ v)
            y = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            y = rearrange(y, 'b (head c) h w -> b (c head) h w', head=self.num_heads, h=h, w=w)
            y = self.project_out(y)
            return y, attn
        else:
            attn = prev_atns
            v = rearrange(x, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            out = (attn @ v)
            y = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            y = rearrange(y, 'b (head c) h w -> b (c head) h w', head=self.num_heads, h=h, w=w)
            y = self.project_out(y)
            return y


class RSA(nn.Module):
    def __init__(self, channels, num_heads, shifts=1, window_sizes=[4, 8, 12], bias=False):
        super(RSA, self).__init__()
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        if prev_atns is None:
            wsize = self.window_sizes
            x_ = x
            if self.shifts > 0:
                x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
            qkv = self.qkv_dwconv(self.qkv(x_))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
            k = rearrange(k, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
            v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q.transpose(-2, -1) @ k) * self.temperature  # b (h w) (dh dw) (dh dw)
            attn = self.act(attn)
            out = (v @ attn)
            out = rearrange(out, 'b (h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h // wsize, w=w // wsize, dh=wsize,
                            dw=wsize)
            if self.shifts > 0:
                out = torch.roll(out, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
            y = self.project_out(out)
            return y, attn
        else:
            wsize = self.window_sizes
            if self.shifts > 0:
                x = torch.roll(x, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
            atn = prev_atns
            v = rearrange(x, 'b (c) (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
            y_ = (v @ atn)
            y_ = rearrange(y_, 'b (h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h // wsize, w=w // wsize, dh=wsize,
                           dw=wsize)
            if self.shifts > 0:
                y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
            y = self.project_out(y_)
            return y

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(FeedForward, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        # h
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        # v
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        # d
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels)


class WFF(nn.Module):
    def __init__(self, in_channels, blocks=2, attention='se'):
        super(WFF, self).__init__()

        self.channels = in_channels
        self.haar = HaarWavelet(in_channels, grad=False)

        self.l_ResConv = ResBlock(in_channels * 2, in_channels, blocks=blocks, attention=attention)
        self.h_ResConv = ResBlock(in_channels * 3 * 2, in_channels * 3, blocks=blocks, attention=attention)

        self.softmax = nn.Softmax(dim=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, raw_image, secret_image, bottom_image, status=True):

        raw_haar = self.haar(raw_image, rev=False)
        raw_l = raw_haar.narrow(1, 0, self.channels)
        raw_h = raw_haar.narrow(1, self.channels, 3 * self.channels)

        secret_haar = self.haar(secret_image, rev=False)
        secret_l = secret_haar.narrow(1, 0, self.channels)
        secret_h = secret_haar.narrow(1, self.channels, 3 * self.channels)

        l_cat = torch.cat((raw_l, secret_l), dim=1)
        l_fusion = self.l_ResConv(l_cat)

        h_cat = torch.cat((raw_h, secret_h), dim=1)
        h_fusion = self.h_ResConv(h_cat)

        if status:
            l_fusion_weight = self.conv2(self.pool(l_fusion))
            bottom_image = self.conv1(bottom_image)
            bottom_image_weight = self.conv3(self.pool(bottom_image))

            weight = torch.cat([l_fusion_weight, bottom_image_weight], 2)
            weight = self.softmax(torch.sigmoid(weight))

            y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
            y1_weight = torch.unsqueeze(weight[:, :, 1], 2)

            l_fusion = y0_weight * l_fusion + y1_weight * bottom_image

        out_image = self.haar(torch.cat([l_fusion, h_fusion], dim=1), rev=True)

        return out_image
