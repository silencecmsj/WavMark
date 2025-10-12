import torch.nn as nn


class Identity(nn.Module):
	"""
	Identity-mapping noise layer. Does not change the image
	"""

	def __init__(self, args=None):
		super(Identity, self).__init__()

	def forward(self, image_cover, device):
		encoded_image = image_cover[1]
		return encoded_image
