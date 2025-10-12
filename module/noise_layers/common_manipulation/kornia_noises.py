import kornia
import torch.nn as nn


# Kornia based noises


# intensity
class GaussianBlur(nn.Module):

    def __init__(self, args=None, kernel_size=(3,3), sigma=(2,2), p=1):
        super(GaussianBlur, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigma, p=p)

    def forward(self, image_cover, device):
        encoded_image = image_cover[1]
        return self.transform(encoded_image)


class GaussianNoise(nn.Module):

    def __init__(self, args=None, mean=0, std=0.1, p=1):
        super(GaussianNoise, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianNoise(mean=mean, std=std, p=p)

    def forward(self, image_cover, device):
        encoded_image = image_cover[1]
        return self.transform(encoded_image)


class MedianBlur(nn.Module):

    def __init__(self, args=None, kernel_size=(3,3)):
        super(MedianBlur, self).__init__()
        self.transform = kornia.filters.MedianBlur(kernel_size=kernel_size)

    def forward(self, image_cover, device):
        encoded_image = image_cover[1]
        return self.transform(encoded_image)


class Brightness(nn.Module):

    def __init__(self, args=None, brightness=0.5, p=1):
        super(Brightness, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(brightness=brightness, p=p)

    def forward(self, image_cover, device):
        encoded_image = image_cover[1]
        out = (encoded_image + 1 ) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Contrast(nn.Module):

    def __init__(self, args=None, contrast=0.5, p=1):
        super(Contrast, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(contrast=contrast, p=p)

    def forward(self, image_cover, device):
        encoded_image = image_cover[1]
        out = (encoded_image + 1) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Saturation(nn.Module):

    def __init__(self, args=None, saturation=0.5, p=1):
        super(Saturation, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(saturation=saturation, p=p)

    def forward(self, image_cover, device):
        cover_image = image_cover[1]
        out = (cover_image + 1) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Hue(nn.Module):

    def __init__(self, args=None, hue=0.1, p=1):
        super(Hue, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(hue=hue, p=p)

    def forward(self, image_cover, device):
        image = image_cover[1]
        out = (image + 1) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


# geometric
class Rotation(nn.Module):

    def __init__(self, args=None, degrees=180, p=1):
        super(Rotation, self).__init__()
        self.transform = kornia.augmentation.RandomRotation(degrees=degrees, p=p)

    def forward(self, image_cover, device):
        encoded_image = image_cover[1]
        return self.transform(encoded_image)


class Affine(nn.Module):

    def __init__(self, args=None, degrees=0, translate=0.1, scale=[0.7,0.7], shear=30, p=1):
        super(Affine, self).__init__()
        self.transform = kornia.augmentation.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, p=p)

    def forward(self, image_cover, deivce):
        encoded_image = image_cover[1]
        return self.transform(encoded_image)

