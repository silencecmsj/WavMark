import numpy as np
import torch
from torch import nn


class Random_Noise(nn.Module):
    def __init__(self, args):
        super(Random_Noise, self).__init__()
        self.len_layers_C = len(args.noise_layers_C)
        self.len_layers_D = len(args.noise_layers_D)
        self.noise_layers_D = args.noise_layers_D

    def forward(self, args, image_cover):
        raw_image = image_cover[0]
        encoded_image = image_cover[1]
        name = image_cover[2]
        noised_image = torch.zeros_like(raw_image)
        label = []

        for index in range(raw_image.shape[0]):
            if args.status == 0:
                random_noise_layer = np.random.choice(args.noise, 1)[0]
            elif args.status == 1:
                random_noise_layer = args.noise[0]

            noised_image[index] = random_noise_layer([raw_image[index].unsqueeze(0), encoded_image[index].unsqueeze(0), name[index]], args.device).to(args.device)

            if type(random_noise_layer).__name__ + '(args)' in self.noise_layers_D:
                label.append(1)
            else:
                label.append(0)

        return noised_image, label