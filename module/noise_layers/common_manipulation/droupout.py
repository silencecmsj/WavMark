import numpy as np
import torch
import torch.nn as nn


class Dropout(nn.Module):

	def __init__(self, args=None, prob=0.3):
		super(Dropout, self).__init__()
		self.prob = prob

	def forward(self, image_cover, device):
		image, encoded_image = image_cover[0], image_cover[1]
		mask = torch.Tensor(np.random.choice([0.0, 1.0], image.shape[2:], p=[self.prob, 1 - self.prob])).to(image.device)
		mask = mask.expand_as(image)
		output = image * mask + encoded_image * (1 - mask)
		return output

