import os

import kornia
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CHDD
from module.DualMark import DualMark
from utils.utils import AverageMeter, Prepare_logger
from module.noise_layers.common_manipulation.kornia_noises import *
from module.noise_layers.common_manipulation.DifferentialJpeg import DifferentialJpeg
from module.noise_layers.common_manipulation.RealJpeg import RealJpeg
from module.noise_layers.common_manipulation.identity import Identity
from module.noise_layers.common_manipulation.droupout import Dropout
from module.noise_layers.common_manipulation.resize import Resize
from module.noise_layers.common_manipulation.salt_pepper import SaltPepper
from module.noise_layers.common_manipulation.kornia_noises import *
from module.noise_layers.deepfake.Simswap import SimSwap
from module.noise_layers.deepfake.StarGAN import StarGAN

import argparse

parser = argparse.ArgumentParser(description='Defense')

# config para
parser.add_argument('--csv_file_path', default=r'data\FFHQ_128.csv', help='features csv file path')
parser.add_argument('--checkpoint_path', type=str,
    default=r'checkpoints\epoch_256_40.037_0.0021.tar', help='checkpoint path')
parser.add_argument('--batch_size', type=int, default=1, help='number of instances in a batch of data (default: 64)')
parser.add_argument('--status', type=int, default=0, help='train:0, test:1')
parser.add_argument('--message_length', type=int, default=128, help='message length')
parser.add_argument('--resolution', type=str, default='256', help='image size')
parser.add_argument('--secret_image_path', type=str,
                    default=r'data\secret_image\BlackWhite256.jpg', help='path')
parser.add_argument('--noise_layers_C', type=list,
                    default=[
                                   #'Identity(args)',
                                    #'Dropout(args)',
                                    #'DifferentialJpeg(args)',
                                    #'Resize(args)',
                                    #'GaussianBlur(args)',
                                     #'MedianBlur(args)',
                                    #'Brightness(args)',
                                    #'Contrast(args)',
                                    #'Saturation(args)',
                                    #'Hue(args)',
                                    #'SaltPepper(args)',
                                   #'GaussianNoise(args)'
                             ], help='weight')
parser.add_argument('--noise_layers_D', type=list, default=[ #'Identity(args)',
                                                                          #'AttGAN1(args)',
                                                                          'StarGAN(args)',
                                                                           #'HiSD(args)',
                                                                          #'SimSwap(args)',
                                                                           #'InfoSwap(args)',
                                                                          #'UniFaceSwap(args)',
                                                                          #'UniFaceReen(args)',
                                                                          #'HyperReenact(args)',
                                                                          #'DiffAE(args)',
                                                                          #'DiffusionClip(args)'
                                                                    ], help='weight')


parser.add_argument('--dataset', default="CelebAHQ", type=str, help='dataset Name')
parser.add_argument('--epoch', type=int, default=200, help='maximum iteration number (default: 100)')
parser.add_argument('--gpus', default="cuda:0", type=str, help='gpus')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
#parser.add_argument('--weight', type=float, default=0.2, help='weight_decay (default: 0.2)')
parser.add_argument('--workers', type=int, default=0, help='number of workers in dataloader')
parser.add_argument('--attention_encoder', type=str, default='se', help='attention layer')
parser.add_argument('--attention_decoder', type=str, default='se', help='attention layer')
parser.add_argument('--beta', type=float, default=0.5, help='message length')
parser.add_argument('--weight', type=list, default=[1, 1, 1], help='weight')
parser.add_argument('--message_range', type=list, default=[0, 1], help='weight')
parser.add_argument(
    "--faceparse_weight",
    type=str,
    default="../../model/faceparse/weights/resnet18.pt",
    help="path to trained model, i.e resnet18/34"
)
# --- StarGAN ---
parser.add_argument('--stargan_path', type=str,
    default=r"module\pretrained_models\StarGAN\models\200000-G.ckpt")
# --- SimSwap ---
parser.add_argument('--simswap_ckpt', type=str, default=r'pretrained_models\SimSwap\G_simswap.pth')
parser.add_argument('--simswap_arcface_ckpt', type=str, default=r'pretrained_models\SimSwap\arcface.pth')

def test(args, model, test_loader):
    model.encoder_decoder.eval()
    step = 0
    total_g_loss_on_encoder = AverageMeter('2', ':.4f')
    total_g_loss_on_decoder = AverageMeter('3', ':.4f')
    total_error_rate = AverageMeter('4', ':.4f')
    total_PSNR = AverageMeter('5', ':.4f')
    total_SSIM = AverageMeter('6', ':.4f')
    total_LPIPS = AverageMeter('7', ':.4f')

    with ((torch.no_grad())):
        for test_data in tqdm(test_loader):
            step += 1
            raw_images, secret_images, name = test_data
            messages = torch.Tensor(
                np.random.choice([0, 1], (raw_images.shape[0], args.message_length))).to(args.device)
            raw_images, secret_images = raw_images.to(args.device), secret_images.to(args.device)
            encoded_images, noised_images, gap, decoded_secret_images, decoded_messages = \
                                model.encoder_decoder(args, raw_images, secret_images, messages, name)

            '''
            decoded message error rate /Dual
            '''
            error_rate = model.decoded_message_error_rate_batch(messages, decoded_messages)
            psnr = -kornia.losses.psnr_loss((encoded_images.detach() + 1) / 2, (raw_images + 1) / 2, 1)
            ssim = 1 - 2 * kornia.losses.ssim_loss((encoded_images.detach() + 1) / 2, (raw_images + 1) / 2, window_size=11, reduction="mean")
            lpips = torch.mean(model.criterion_LPIPS((encoded_images.detach() + 1) / 2, (raw_images + 1) / 2))

            total_error_rate.update(error_rate, 1)
            total_g_loss_on_encoder.update(g_loss_on_encoder.item(), 1)
            total_g_loss_on_decoder.update(g_loss_on_decoder.item(), 1)
            total_PSNR.update(psnr.item(), 1)
            total_SSIM.update(ssim.item(), 1)
            total_LPIPS.update(lpips.item(), 1)

            raw_images = (raw_images + 1) / 2
            encoded_images = (encoded_images + 1) / 2
            noised_images = (noised_images + 1) / 2
            decoded_secret_images = (decoded_secret_images + 1) / 2
            if len(args.noise_layers_C) != 0 :
                path = 'common/'
                method = args.noise_layers_C[0].split('(')[0] + '/'
            else:
                path = 'deepfake/'
                method = args.noise_layers_D[0].split('(')[0] + '/'

            path1 = r'results/' + args.dataset + '/' + args.resolution + '/' + path + method + '/OI/'
            if not os.path.exists(path1):
                os.makedirs(path1)
            torchvision.utils.save_image(raw_images, path1 + str(step) + '.png', nrow=1)
            path4 = r'results/' + args.dataset + '/' + args.resolution + '/' + path + method + '/EI/'
            if not os.path.exists(path4):
                os.makedirs(path4)
            torchvision.utils.save_image(encoded_images, path4 + str(step) + '.png', nrow=1)
            #torchvision.utils.save_image(encoded_images, 'results/' + args.dataset + '/' + args.resolution + '/' + path + method + '/EI/' + str(step) + '.png', nrow=1)
            path2 = r'results/' + args.dataset + '/' + args.resolution + '/' + path + method + '/NI/'
            if not os.path.exists(path2):
                os.makedirs(path2)
            torchvision.utils.save_image(noised_images, path2 + str(step) + '.png', nrow=1)
            path3 = r'E:/PythonProject/Dualmark/results/' + args.dataset + '/' + args.resolution + '/' + path + method + '/SI/'
            if not os.path.exists(path3):
                os.makedirs(path3)
            torchvision.utils.save_image(decoded_secret_images, 'results/' + args.dataset + '/' + args.resolution + '/' + path + method + '/SI/' + str(step) + '.png', nrow=1)

            image_save = torch.cat([raw_images[i],
                                        encoded_images[:1], noised_images[:1], decoded_secret_images[:1]], dim=0)

            torchvision.utils.save_image(image_save, 'results/' + str(step) + '.png', nrow=8)

        result = {
            "BCR": (1 - total_error_rate.avg) * 100,
            "g_loss_on_encoder": total_g_loss_on_encoder.avg,
            "g_loss_on_decoder": total_g_loss_on_decoder.avg,
            "PSNR": total_PSNR.avg,
            "SSIM": total_SSIM.avg,
            "LPIPS": total_LPIPS.avg,
        }
    return result

if __name__ == '__main__':

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger = Prepare_logger(eval=False)
    logger.info(f'Loading model.')
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    test_dataset = CHDD(args, 2, transforms=data_transforms)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    logger.info('TestSet Number:{}'.format(test_dataset.num))

    checkpoint = torch.load(args.checkpoint_path)
    model = DualMark(args, args.lr, args.beta, args.weight)
    model.encoder_decoder.load_state_dict(checkpoint['encoder_decoder'])
    # noise layer
    layers = args.noise_layers_C + args.noise_layers_D
    for i in range(len(layers)):
        layers[i] = eval(layers[i])
    args.noise = nn.Sequential(*layers)

    test_result = test(args, model, test_loader)

    logger.info('{} {}'.format(args.noise_layers_C, args.noise_layers_D))
    logger.info('Test...\n'
                'BCR:                               {:.4f}\n'
                'g_loss_on_encoder:                 {:.4f}\n'
                'g_loss_on_decoder:                 {:.4f}\n'
                'SSIM:                              {:.4f}\n'
                'PSNR:                              {:.4f}\n'
                'LPIPS:                              {:.4f}\n'
                .format(test_result['BCR'],
                        test_result['g_loss_on_encoder'],
                        test_result['g_loss_on_decoder'],
                        test_result['SSIM'],
                        test_result['PSNR'],
                        test_result['LPIPS']))