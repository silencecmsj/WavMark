import torch
import torch
import torch.nn as nn
from torchvision import transforms

from models.diff_jpeg import DiffJPEGCoding


#(72, 91)
class DifferentialJpeg(nn.Module):
    """
    Drops random pixels from the noised image and substitues them with the pixels from the cover image
    """
    def __init__(self, args=None):
        super(DifferentialJpeg, self).__init__()
        self.ratio = [50, 51]
        self.diff_jpeg = DiffJPEGCoding()
        self.transforms = transforms.Compose(
            [
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def forward(self, image_cover, device):

        encoded_image = (image_cover[1] + 1) / 2 * 255
        ratio = torch.randint(self.ratio[0], self.ratio[1], (1,)).to(device)
        noise_image = self.diff_jpeg(image_rgb=encoded_image, jpeg_quality=ratio).to(device)
        noise_image = noise_image / 255 * 2 - 1

        return noise_image