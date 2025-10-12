import cv2
import lpips
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from kornia import metrics

# Initialize the LPIPS model using a pretrained network (e.g., VGG). Set to GPU if available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lpips_model = lpips.LPIPS(net='vgg').to(device)

def calculate_lpips(image1, image2):
    def load_image(image_path):
        """
        Load an image and convert it to tensor.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),          # Convert the image to a torch tensor
            transforms.Normalize(mean=[0.5, 5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)  # Add batch dimension

    # Load your images
    image1 = load_image(image1).to(device)
    image2 = load_image(image2).to(device)

    # Calculate LPIPS
    distance = lpips_model(image1, image2)
    return distance.item()

import torch.nn.functional as F

def calculate_l2_distance(img_path1, img_path2):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')

    tensor1 = transform(img1)
    tensor2 = transform(img2)

    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Image shapes are different: {tensor1.shape} vs {tensor2.shape}")

    mse = F.mse_loss(tensor1, tensor2)

    return mse.item()

def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ]
            )
    return transform(image).unsqueeze(0)

def calculate_ssim_kornia(img_path1, img_path2):
    img1 = load_image_as_tensor(img_path1)
    img2 = load_image_as_tensor(img_path2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1, img2 = img1.to(device), img2.to(device)

    ssim_map = metrics.ssim(img1, img2, window_size=11, max_val=1.0)
    return ssim_map.mean().item()

def calculate_psnr_kornia(img_path1, img_path2):
    img1 = load_image_as_tensor(img_path1)
    img2 = load_image_as_tensor(img_path2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1, img2 = img1.to(device), img2.to(device)

    psnr_val = metrics.psnr(img1, img2, max_val=1.0)
    return psnr_val.item()