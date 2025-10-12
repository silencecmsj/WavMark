import random

import torch
import torch.nn as nn
import torchvision
from PIL import Image

from torchvision import transforms

import option
from module.noise_layers.deepfake.stargan.stargan import Generator


class StarGAN(nn.Module):
    def __init__(self, args):
        super(StarGAN, self).__init__()
        self.stargan_path = args.stargan_path
        # Black_Hair Blond_Hair Brown_Hair Male Young
        self.label = torch.tensor([
            [1., 0., 0., 0., 1.],  #
            [0., 1., 0., 0., 1.],
            [0., 0., 1., 0., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0.],
        ])

        self.generator = Generator(conv_dim=64, c_dim=5, repeat_num=6).to(args.device)
        self.generator.load_state_dict(torch.load(self.stargan_path, map_location=lambda storage, loc: storage))
        for param in self.generator.parameters():
            param.requires_grad = False

    def forward(self, image_cover, device):
        encoded_image = image_cover[1]
        index = random.choice([0, 1, 2, 3, 4])
        label = self.label[index].unsqueeze(0).clone().to(device)
        noised_image = self.generator(encoded_image, label).to(device)
        return noised_image