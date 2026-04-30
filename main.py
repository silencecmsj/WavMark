import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from visdom import Visdom
from module.noise_layers.common_manipulation.kornia_noises import *
from module.noise_layers.common_manipulation.DifferentialJpeg import DifferentialJpeg
from module.noise_layers.common_manipulation.RealJpeg import RealJpeg
from module.noise_layers.common_manipulation.identity import Identity
from module.noise_layers.common_manipulation.droupout import Dropout
from module.noise_layers.common_manipulation.resize import Resize
from module.noise_layers.common_manipulation.salt_pepper import SaltPepper
from module.noise_layers.common_manipulation.kornia_noises import *
from module.noise_layers.deepfake.Simswap import SimSwap
from module.noise_layers.deepfake.HiSD import HiSD
from module.noise_layers.deepfake.StarGAN import StarGAN
#from module.noise_layers.deepfake.Ganimation import GANimation
from module.noise_layers.deepfake.InfoSwap import InfoSwap
#from module.noise_layers.deepfake.UniFaceSwap import UniFaceSwap
#from module.noise_layers.deepfake.UniFaceReen import UniFaceReen
# from module.noise_layers.deepfake.HyperReenact import HyperReenact
#from module.noise_layers.deepfake.AttentionGAN import AttentionGAN
from module.noise_layers.deepfake.AttGAN import AttGAN1
from module.noise_layers.deepfake.DiffAE import DiffAE
from module.noise_layers.deepfake.DiffusionClip import DiffusionClip

import option
from dataset import CHDD
from module.DualMark import DualMark
from test import test
from train import train
from utils.utils import Prepare_logger
from validate import validate
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = option.parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    logger = Prepare_logger(eval=False)
    logger.info(f'Loading model.')

    viz = Visdom()
    #viz.text("This is environment 1", env="mydata")

    data_transforms = transforms.Compose(
                [
                    #transforms.Resize(128),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
    model = DualMark(args, args.lr, args.beta, args.weight)
    checkpoint = torch.load(args.checkpoint_path)
    model.encoder_decoder.load_state_dict(checkpoint['encoder_decoder'])
    # noise layer
    layers = args.noise_layers_C + args.noise_layers_D
    for i in range(len(layers)):
        layers[i] = eval(layers[i])
    args.noise = nn.Sequential(*layers)

    if args.status == 0:
        train_dataset = CHDD(args, 0, transforms=data_transforms)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
        logger.info('TrainSet Number:{}'.format(train_dataset.num))

        validate_dataset = CHDD(args, 1, transforms=data_transforms)
        validate_loader = DataLoader(validate_dataset,
                                 batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.workers, pin_memory=True)
        logger.info('ValidateSet Number:{}'.format(validate_dataset.num))

        result = None
        start_epoch = 0
        max_psnr = 0
        min_error_rate = 1
        for epoch in range(start_epoch + 1, args.epoch + 1):
            logger.info('Epoch: {}/{}'.format(epoch, args.epoch))

            train_result = train(args, model, train_loader)

            if epoch == 1:
                viz.line([[train_result['error_rate'], train_result['g_loss_on_encoder'], train_result['g_loss_on_encoder_perceptual'],
                                        train_result['g_loss_on_decoder1'],  train_result['g_loss_on_decoder2']]],  ## Y的第一个点坐标
                         [1.],
                         win="Train Loss",
                         opts=dict(title='Train LOSS', legend=['Error Rate', 'Encoder Loss', 'Perceptual Loss','Decoder1 Loss', 'Decoder2 Loss'])  ## 图像标例
                         )
                viz.line([[train_result['error_rate'], train_result['g_loss_on_encoder'],
                           train_result['g_loss_on_encoder_perceptual'],
                           train_result['g_loss_on_decoder1'],  train_result['g_loss_on_decoder2']]],  ## Y的下一个点
                         [epoch],
                         win="Train Loss",
                         update='append'
                         )
            else:
                viz.line([[      train_result['error_rate'], train_result['g_loss_on_encoder'], train_result['g_loss_on_encoder_perceptual'],
                                        train_result['g_loss_on_decoder1'],  train_result['g_loss_on_decoder2']]],  ## Y的下一个点
                         [epoch],
                         win="Train Loss",
                         update='append'
                         )

            logger.info('Train...\n'
                'error_rate:                        {:.4f}\n'
                'd_loss:                            {:.4f}\n'
                'g_loss_on_encoder:                 {:.4f}\n'
                'g_loss_on_encoder_perceptual:      {:.4f}\n'
                'g_loss_on_discriminator:           {:.4f}\n'
                'g_loss_on_decoder1:                {:.4f}\n'
                'g_loss_on_decoder2:                {:.4f}\n'
                'SSIM:                              {:.4f}\n'
                'PSNR:                              {:.4f}\n'
                .format(train_result['error_rate'],
                                train_result['d_loss'],
                                train_result['g_loss_on_encoder'],
                                train_result['g_loss_on_encoder_perceptual'],
                                train_result['g_loss_on_discriminator'],
                                train_result['g_loss_on_decoder1'],
                                train_result['g_loss_on_decoder2'],
                                train_result['SSIM'],
                                train_result['PSNR']))

            validate_result = validate(args, model, validate_loader, epoch)

            if epoch == 1:
                viz.line([[validate_result['error_rate'], validate_result['g_loss_on_encoder'], validate_result['g_loss_on_decoder']]],  ## Y的第一个点坐标
                         [1.],
                         win="Validate Loss",
                         opts=dict(title='Validate Loss', legend=['Error Rate', 'Encoder Loss', 'Decoder Loss', 'SSIM'])  ## 图像标例
                         )
                viz.line([[validate_result['error_rate'], validate_result['g_loss_on_encoder'], validate_result['g_loss_on_decoder']]],  ## Y的下一个点
                         [epoch],
                         win="Validate Loss",
                         update='append'
                         )
            else:
                viz.line([[validate_result['error_rate'], validate_result['g_loss_on_encoder'], validate_result['g_loss_on_decoder']]],  ## Y的下一个点
                         [epoch],
                         win="Validate Loss",
                         update='append'
                         )

            if epoch == 1:
                viz.line([[train_result['PSNR'], validate_result['PSNR'], 100 * train_result['SSIM'],
                           100 * validate_result['SSIM']]],
                         [1.],
                         win="Train&Validate PSNR SSIM",
                         opts=dict(title='Train&Validate PSNR', legend=['TPSNR', 'VPSNR', 'TSSIM', 'VSSIM'])
                         )

            viz.line([[train_result['PSNR'], validate_result['PSNR'], 100 * train_result['SSIM'],
                       100 * validate_result['SSIM']]],
                     [epoch],
                     win="Train&Validate PSNR SSIM",
                     update='append'
                     )

            logger.info('Validate...\n'
                        'error_rate:                        {:.4f}\n'
                        'g_loss_on_encoder:                 {:.4f}\n'
                        'g_loss_on_decoder:                 {:.4f}\n'
                        'SSIM:                              {:.4f}\n'
                        'PSNR:                              {:.4f}\n'
                        .format(validate_result['error_rate'],
                                validate_result['g_loss_on_encoder'],
                                validate_result['g_loss_on_decoder'],
                                validate_result['SSIM'],
                                validate_result['PSNR']))

            if 38 < validate_result['PSNR']:
                max_psnr = validate_result['PSNR']
                min_error_rate = validate_result['error_rate']
                torch.save(
                    {
                        'epoch': epoch,
                        'encoder_decoder': model.encoder_decoder.state_dict(),
                        #'decoder': model.encoder_decoder.decoder.state_dict(),
                        'optimizer': model.opt_encoder_decoder.state_dict(),
                    }, './checkpoints/epoch_' + str(epoch) + '_' + str(max_psnr)[:6] + '_' + str(min_error_rate)[:6] + '.tar'
                )
            viz.save(['main'])
        os.system("shutdown -s -t 1 ")
    else:
        test_dataset = CHDD(args, 2, transforms=data_transforms)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
        logger.info('TestSet Number:{}'.format(test_dataset.num))

        checkpoint = torch.load(args.checkpoint_path)
        model.encoder_decoder.load_state_dict(checkpoint['encoder_decoder'])
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