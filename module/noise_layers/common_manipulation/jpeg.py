import os
import random
import string

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from module.diff_jpeg import DiffJPEGCoding


class JpegTest(nn.Module):
	def __init__(self, args=None, Q=50, subsample=2, path="temp/"):
		super(JpegTest, self).__init__()
		self.Q = Q
		self.subsample = subsample
		self.path = path
		if not os.path.exists(path): os.mkdir(path)
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])

	def get_path(self):
		return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg"

	def forward(self, image_cover_mask):
		image, cover_image = image_cover_mask[0], image_cover_mask[1]

		noised_image = torch.zeros_like(image)

		for i in range(image.shape[0]):
			single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
			im = Image.fromarray(single_image)

			file = self.get_path()
			while os.path.exists(file):
				file = self.get_path()
			im.save(file, format="JPEG", quality=self.Q, subsampling=self.subsample)
			jpeg = np.array(Image.open(file), dtype=np.uint8)
			os.remove(file)

			noised_image[i] = self.transform(jpeg).unsqueeze(0).to(image.device)

		return noised_image

#(72, 91)
class Jpeg(nn.Module):
    """
    Drops random pixels from the noised image and substitues them with the pixels from the cover image
    """
    def __init__(self):
        super(Jpeg, self).__init__()
        self.ratio = [50, 50]
        self.diff_jpeg = DiffJPEGCoding()
        self.transforms = transforms.Compose(
            [
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def forward(self, image):
        protect_image = (image[0] + 1) / 2 * 255
        ori_image = (image[1] + 1) / 2 * 255
        ratio = torch.randint(self.ratio[0], self.ratio[1], (1,))
        enhance_protect_image = self.diff_jpeg(image_rgb=protect_image, jpeg_quality=ratio)
        enhance_protect_image = enhance_protect_image / 255 * 2 - 1
        enhance_ori_image = self.diff_jpeg(image_rgb=ori_image, jpeg_quality=ratio)
        enhance_ori_image = enhance_ori_image / 255 * 2 - 1
        return enhance_protect_image, enhance_ori_image