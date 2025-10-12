import numpy as np
import torch
import torch.nn as nn
import torchvision
#
#
# class SaltPepper(nn.Module):
#
# 	def __init__(self, prob=0.1):
# 		super(SaltPepper, self).__init__()
# 		self.prob = prob
#
# 	def sp_noise(self, image, prob):
# 		mask = torch.Tensor(np.random.choice((0, 1, 2), image.shape[2:], p=[1 - prob, prob / 2., prob / 2.]))
# 		mask = mask.expand_as(image)
#
# 		image[mask == 1] = 1  # salt
# 		image[mask == 2] = -1  # pepper
#
# 		return image
#
# 	def forward(self, image_cover, device):
# 		cover_image = image_cover[1]
# 		noised_image = self.sp_noise(cover_image, self.prob) #image * mask + self.sp_noise(image, self.prob) * (1 - mask)
# 		#mask = mask[:, 0: 3, :, :]
# 		return noised_image

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms


class SaltPepper(nn.Module):
    def __init__(self, args=None, prob=0.1):
        super(SaltPepper, self).__init__()
        self.prob = prob

    def sp_noise(self, image, prob):

        device = image.device

        rand_tensor = torch.rand(image.shape, device=device)


        salt_mask = (rand_tensor < prob / 2).float()
        pepper_mask = (rand_tensor >= 1 - prob / 2).float()

        noised_image = image * (1 - salt_mask - pepper_mask) + salt_mask * 1.0 + pepper_mask * -1.0

        return noised_image

    def forward(self, image_cover, device=None):

        encoded_image = image_cover[1]
        noised_image = self.sp_noise(encoded_image, self.prob)
        return noised_image

