import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from module.ConvBlock import ConvBlock
from module.ResBlock import ResBlock
from module.WaveletTransform import HaarWavelet, WFF

class Encoder_Message(nn.Module):

    def __init__(self, message_length=64, blocks=2, channels=64, attention=None):
        super(Encoder_Message, self).__init__()

        self.message_length = message_length

        # raw down
        self.raw_conv1 = ConvBlock(3, 16, blocks=blocks)
        self.raw_down1 = Down(16, 32, blocks=blocks)
        self.raw_down2 = Down(32, 64, blocks=blocks)
        self.raw_down3 = Down(64, 128, blocks=blocks)
        self.raw_down4 = Down(128, 256, blocks=blocks)

        # secret down
        self.secret_conv1 = ConvBlock(3, 16, blocks=blocks)
        self.secret_down1 = Down(16, 32, blocks=blocks)
        self.secret_down2 = Down(32, 64, blocks=blocks)
        self.secret_down3 = Down(64, 128, blocks=blocks)
        self.secret_down4 = Down(128, 256, blocks=blocks)

        self.linear0 = nn.Linear(message_length, message_length * message_length)
        self.linear1 = nn.Linear(message_length, message_length * message_length)
        self.linear2 = nn.Linear(message_length, message_length * message_length)
        self.linear3 = nn.Linear(message_length, message_length * message_length)
        self.linear4 = nn.Linear(message_length, message_length * message_length)

        self.Conv_message0 = ConvBlock(1, 16, blocks=blocks)
        self.Conv_message1 = ConvBlock(1, 32, blocks=blocks)
        self.Conv_message2 = ConvBlock(1, 64, blocks=blocks)
        self.Conv_message3 = ConvBlock(1, 128, blocks=blocks)
        self.Conv_message4 = ConvBlock(1, 256, blocks=blocks)

        self.res0 = ResBlock(16 * 3, 16, blocks=blocks, attention=attention)
        self.res1 = ResBlock(32 * 3, 32, blocks=blocks, attention=attention)
        self.res2 = ResBlock(64 * 3, 64, blocks=blocks, attention=attention)
        self.res3 = ResBlock(128 * 3, 128, blocks=blocks, attention=attention)
        self.res4 = ResBlock(256 * 3, 256, blocks=blocks, attention=attention)

        self.wff0 = WFF(16)
        self.wff1 = WFF(32)
        self.wff2 = WFF(64)
        self.wff3 = WFF(128)
        self.wff4 = WFF(256)

        self.conv = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, raw_image, secret_image, message):

        # raw image
        raw_d0 = self.raw_conv1(raw_image) # B,16,H,W
        raw_d1 = self.raw_down1(raw_d0) # B,32,H/2,W/2
        raw_d2 = self.raw_down2(raw_d1) # B,64,H/4,W/4
        raw_d3 = self.raw_down3(raw_d2) # B,128,H/8,W/8
        raw_d4 = self.raw_down4(raw_d3) # B,256,H/16,W/16

        # secret image
        secret_d0 = self.secret_conv1(secret_image)
        secret_d1 = self.secret_down1(secret_d0)
        secret_d2 = self.secret_down2(secret_d1)
        secret_d3 = self.secret_down3(secret_d2)
        secret_d4 = self.secret_down4(secret_d3)

        expanded_message = self.linear4(message)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(raw_d4.shape[2], raw_d4.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message4(expanded_message)

        wff4 = self.wff4(raw_d4, expanded_message, None, False)
        wff4 = torch.cat((raw_d4, wff4, secret_d4), dim=1)
        wff4 = self.res4(wff4)

        expanded_message = self.linear3(message)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(raw_d3.shape[2], raw_d3.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message3(expanded_message)

        wff3 = self.wff3(raw_d3, expanded_message, wff4, True)
        wff3 = torch.cat((raw_d3, wff3, secret_d3), dim=1)
        wff3 = self.res3(wff3)

        expanded_message = self.linear2(message)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(raw_d2.shape[2], raw_d2.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message2(expanded_message)

        wff2 = self.wff2(raw_d2, expanded_message, wff3, True)
        wff2 = torch.cat((raw_d2, wff2, secret_d2), dim=1)
        wff2 = self.res2(wff2)


        expanded_message = self.linear1(message)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(raw_d1.shape[2], raw_d1.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message1(expanded_message)

        wff1 = self.wff1(raw_d1, expanded_message, wff2, True)
        wff1 = torch.cat((raw_d1, wff1, secret_d1), dim=1)
        wff1 = self.res1(wff1)


        expanded_message = self.linear0(message)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(raw_d0.shape[2], raw_d0.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message0(expanded_message)

        wff0 = self.wff0(raw_d0, expanded_message, wff1, True)
        wff0 = torch.cat((raw_d0, wff0, secret_d0), dim=1)
        wff0 = self.res0(wff0)

        image = self.conv(torch.cat((raw_image, wff0), dim=1))

        return image

class Encoder(nn.Module):

    def __init__(self, message_length=64, blocks=2, channels=64, attention=None):
        super(Encoder, self).__init__()

        self.message_length = message_length

        # raw down
        self.raw_conv1 = ConvBlock(3, 16, blocks=blocks)
        self.raw_down1 = Down(16, 32, blocks=blocks)
        self.raw_down2 = Down(32, 64, blocks=blocks)
        self.raw_down3 = Down(64, 128, blocks=blocks)
        self.raw_down4 = Down(128, 256, blocks=blocks)

        # secret down
        self.secret_conv1 = ConvBlock(3, 16, blocks=blocks)
        self.secret_down1 = Down(16, 32, blocks=blocks)
        self.secret_down2 = Down(32, 64, blocks=blocks)
        self.secret_down3 = Down(64, 128, blocks=blocks)
        self.secret_down4 = Down(128, 256, blocks=blocks)

        self.linear0 = nn.Linear(message_length, message_length * message_length)
        self.linear1 = nn.Linear(message_length, message_length * message_length)
        self.linear2 = nn.Linear(message_length, message_length * message_length)
        self.linear3 = nn.Linear(message_length, message_length * message_length)
        self.linear4 = nn.Linear(message_length, message_length * message_length)

        self.Conv_message0 = ConvBlock(1, channels, blocks=blocks)
        self.Conv_message1 = ConvBlock(1, channels, blocks=blocks)
        self.Conv_message2 = ConvBlock(1, channels, blocks=blocks)
        self.Conv_message3 = ConvBlock(1, channels, blocks=blocks)
        self.Conv_message4 = ConvBlock(1, channels, blocks=blocks)

        self.res0 = ResBlock(16 * 2 + channels, 16, blocks=blocks, attention=attention)
        self.res1 = ResBlock(32 * 2 + channels, 32, blocks=blocks, attention=attention)
        self.res2 = ResBlock(64 * 2 + channels, 64, blocks=blocks, attention=attention)
        self.res3 = ResBlock(128 * 2 + channels, 128, blocks=blocks, attention=attention)
        self.res4 = ResBlock(256 * 2 + channels, 256, blocks=blocks, attention=attention)

        self.wff0 = WFF(16)
        self.wff1 = WFF(32)
        self.wff2 = WFF(64)
        self.wff3 = WFF(128)
        self.wff4 = WFF(256)

        self.conv = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, raw_image, secret_image, message):

        # raw image
        raw_d0 = self.raw_conv1(raw_image) # B,16,H,W
        raw_d1 = self.raw_down1(raw_d0) # B,32,H/2,W/2
        raw_d2 = self.raw_down2(raw_d1) # B,64,H/4,W/4
        raw_d3 = self.raw_down3(raw_d2) # B,128,H/8,W/8
        raw_d4 = self.raw_down4(raw_d3) # B,256,H/16,W/16

        # secret image
        secret_d0 = self.secret_conv1(secret_image)
        secret_d1 = self.secret_down1(secret_d0)
        secret_d2 = self.secret_down2(secret_d1)
        secret_d3 = self.secret_down3(secret_d2)
        secret_d4 = self.secret_down4(secret_d3)

        wff4 = self.wff4(raw_d4, secret_d4, None, False)
        expanded_message = self.linear4(message)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(wff4.shape[2], wff4.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message4(expanded_message)
        wff4 = torch.cat((raw_d4, wff4, expanded_message), dim=1)
        wff4 = self.res4(wff4)

        wff3 = self.wff3(raw_d3, secret_d3, wff4, True)
        expanded_message = self.linear3(message)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(wff3.shape[2], wff3.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message3(expanded_message)
        wff3 = torch.cat((raw_d3, wff3, expanded_message), dim=1)
        wff3 = self.res3(wff3)

        wff2 = self.wff2(raw_d2, secret_d2, wff3, True)
        expanded_message = self.linear2(message)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(wff2.shape[2], wff2.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message2(expanded_message)
        wff2 = torch.cat((raw_d2, wff2, expanded_message), dim=1)
        wff2 = self.res2(wff2)

        wff1 = self.wff1(raw_d1, secret_d1, wff2, True)
        expanded_message = self.linear1(message)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(wff1.shape[2], wff1.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message1(expanded_message)
        wff1 = torch.cat((raw_d1, wff1, expanded_message), dim=1)
        wff1 = self.res1(wff1)

        wff0 = self.wff0(raw_d0, secret_d0, wff1, True)
        expanded_message = self.linear0(message)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(wff0.shape[2], wff0.shape[3]),
                                         mode='nearest')
        expanded_message = self.Conv_message0(expanded_message)
        wff0 = torch.cat((raw_d0, wff0, expanded_message), dim=1)
        wff0 = self.res0(wff0)

        image = self.conv(torch.cat((raw_image, wff0), dim=1))

        return image


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(Down, self).__init__()
        self.layer = torch.nn.Sequential(
            ConvBlock(in_channels, in_channels, stride=2),
            ConvBlock(in_channels, out_channels, blocks=blocks)
        )

    def forward(self, x):
        return self.layer(x)