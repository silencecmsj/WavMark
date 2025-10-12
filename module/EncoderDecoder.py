import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d

from module.ConvBlock import ConvBlock
from module.Decoder import Decoder
from module.Encoder import Encoder
from module.Random_Noise import Random_Noise


class EncoderDecoder(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, args):
		super(EncoderDecoder, self).__init__()
		self.encoder = Encoder(args.message_length, attention = 'se')
		self.noise = Random_Noise(args)
		self.decoder = Decoder(args.message_length, attention = 'se')

	def forward(self, args, raw_images, secret_images, messages, name):

		encoded_images = self.encoder(raw_images, secret_images, messages)
		noised_image, label = self.noise(args, [raw_images, encoded_images, name])

		decoded_images, decoded_messages = self.decoder(noised_image, noised_image)

		#decoded_clean_images, decoded_clean_messages = self.decoder(raw_images, raw_images)

		return encoded_images, noised_image, label, decoded_images, decoded_messages#, decoded_clean_images