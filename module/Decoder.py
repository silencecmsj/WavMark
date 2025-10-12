import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from module.ConvBlock import ConvBlock
from module.ResBlock import ResBlock, BottleneckBlock
from module.WaveletTransform import HaarWavelet, WFF


class Decoder(nn.Module):

	def __init__(self, message_length=0, blocks=2, channels=64, attention=None):
		super(Decoder, self).__init__()
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

		self.wff0 = WFF(16)
		self.wff1 = WFF(32)
		self.wff2 = WFF(64)
		self.wff3 = WFF(128)
		self.wff4 = WFF(256)

		self.conv_image = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
		self.conv_message = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
		self.message_layer = nn.Linear(message_length * message_length, message_length)

	def forward(self, raw_image, secret_image):

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
		wff3 = self.wff3(raw_d3, secret_d3, wff4, True)
		wff2 = self.wff2(raw_d2, secret_d2, wff3, True)
		wff1 = self.wff1(raw_d1, secret_d1, wff2, True)
		wff0 = self.wff0(raw_d0, secret_d0, wff1, True)

		image = self.conv_image(wff0)
		message = self.conv_message(wff0)

		message = F.interpolate(message, size=(self.message_length, self.message_length),
								mode='nearest')
		message = message.view(message.shape[0], -1)
		message = torch.sigmoid(self.message_layer(message))

		return image, message

class Decoder_Message(nn.Module):

	def __init__(self, message_length=0, blocks=2, channels=64, attention=None):
		super(Decoder_Message, self).__init__()
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

		self.wff0 = WFF(16)
		self.wff1 = WFF(32)
		self.wff2 = WFF(64)
		self.wff3 = WFF(128)
		self.wff4 = WFF(256)

		self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
		self.message_layer = nn.Linear(message_length * message_length, message_length)

	def forward(self, raw_image, secret_image):

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
		wff3 = self.wff3(raw_d3, secret_d3, wff4, True)
		wff2 = self.wff2(raw_d2, secret_d2, wff3, True)
		wff1 = self.wff1(raw_d1, secret_d1, wff2, True)
		wff0 = self.wff0(raw_d0, secret_d0, wff1, True)

		image = self.conv(wff0)

		message = F.interpolate(image, size=(self.message_length, self.message_length),
								mode='nearest')
		message = message.view(message.shape[0], -1)
		message = torch.sigmoid(self.message_layer(message))

		return message





class SENet_decoder(nn.Module):
	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=2):
		super(SENet_decoder, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, 1)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, 1)
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate)
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

class Decoder_Message1(nn.Module):

	def __init__(self, message_length=64, blocks=3, channels=64):
		super(Decoder_Message1, self).__init__()

		self.first_layers = nn.Sequential(
			ConvBlock(3, channels),
			SENet_decoder(channels, channels, blocks=blocks + 1),
			ConvBlock(channels * (2 ** blocks), channels),
		)
		self.keep_layers = ResBlock(channels, channels, blocks=1)

		self.final_layer = ConvBlock(channels, 1)

		self.message_layer = nn.Linear(256, message_length)

	def forward(self, encoded_image):

		x = self.first_layers(encoded_image)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)

		decoded_message = torch.sigmoid(self.message_layer(x))

		return decoded_message


class Down(nn.Module):
	def __init__(self, in_channels, out_channels, blocks):
		super(Down, self).__init__()
		self.layer = torch.nn.Sequential(
			ConvBlock(in_channels, in_channels, stride=2),
			ConvBlock(in_channels, out_channels, blocks=blocks)
		)

	def forward(self, x):
		return self.layer(x)



if __name__ == '__main__':

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	image1 = torch.rand((32, 3, 128, 128)).to(device)
	image2 = torch.rand((32, 3, 128, 128)).to(device)

	model = Decoder_Message().to(device)

	feature = model(image1)

	print(feature.shape)