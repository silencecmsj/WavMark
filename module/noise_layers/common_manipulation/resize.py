import torch.nn as nn
import torch.nn.functional as F


class Resize(nn.Module):
    """
    Resize the image.
    """
    def __init__(self, args=None, down_scale=0.5):
        super(Resize, self).__init__()
        self.down_scale = down_scale

    def forward(self, image_cover, device):
        encoded_image = image_cover[1]
        #
        noised_down = F.interpolate(
                                    encoded_image,
                                    size=(int(self.down_scale * encoded_image.shape[2]), int(self.down_scale * cover_image.shape[3])),
                                    mode='nearest'
                                    )
        noised_up = F.interpolate(
                                    noised_down,
                                    size=(encoded_image.shape[2], encoded_image.shape[3]),
                                    mode='nearest'
                                    )

        return noised_up


