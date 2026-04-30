import kornia
import numpy as np
import torch
from tqdm import tqdm

from utils.utils import AverageMeter

def train(args, model, train_loader):
    model.encoder_decoder.train()
    model.discriminator.train()
    total_g_loss_on_encoder = AverageMeter('2', ':.4f')
    total_g_loss_on_decoder1 = AverageMeter('3', ':.4f')
    total_g_loss_on_decoder2 = AverageMeter('1', ':.4f')
    total_g_loss_on_decoder3 = AverageMeter('10', ':.4f')
    total_error_rate = AverageMeter('4', ':.4f')
    total_PSNR = AverageMeter('5', ':.4f')
    total_SSIM = AverageMeter('6', ':.4f')
    total_g_loss_on_encoder_perceptual  = AverageMeter('7', ':.4f')
    total_g_loss_on_d_loss = AverageMeter('8', ':.4f')
    total_g_loss_on_discriminator = AverageMeter('9', ':.4f')

    with ((torch.enable_grad())):
        for train_data in tqdm(train_loader):

            raw_images, secret_images, name = train_data
            messages = torch.Tensor(
                np.random.choice([0, 1], (raw_images.shape[0], args.message_length))).to(args.device)

            # use device to compute
            raw_images, secret_images = raw_images.to(args.device), secret_images.to(args.device)
            # background_images = images
            encoded_images, _, label, decoded_secret_images, decoded_messages = \
                                            model.encoder_decoder(args, raw_images, secret_images, messages, name)

            '''
            	train discriminator
            '''
            for p in model.discriminator.parameters():
                p.requires_grad = True

            model.opt_discriminator.zero_grad()
            d_label_cover = model.discriminator(raw_images.clone().detach())
            d_label_encoded = model.discriminator(encoded_images.clone().detach())

            d_loss_real = model.criterion_MSE(d_label_cover, torch.ones_like(d_label_cover) * model.label_cover)
            d_loss_fake = model.criterion_MSE(d_label_encoded, torch.zeros_like(d_label_encoded) * model.label_encoded)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            model.opt_discriminator.step()

            '''
                train encoder and decoder
            '''
            for p in model.discriminator.parameters():
                p.requires_grad = False

            model.opt_encoder_decoder.zero_grad()
            g_label_encoded = model.discriminator(encoded_images)

            g_loss_on_discriminator = model.criterion_MSE(g_label_encoded, torch.ones_like(g_label_encoded) * model.label_cover) / 2
            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder = model.criterion_MSE(encoded_images, raw_images.clone().detach())
            g_loss_on_encoder_perceptual = model.lpips_loss(encoded_images, raw_images.clone().detach()).mean()
            # RESULT : the decoded message should be similar to the raw message /
            target_secret_images = secret_images.clone()
            for index, value in enumerate(label):
                if value == 1:
                    target_secret_images[index, :] = 1.0
            g_loss_on_decoder1 = model.criterion_MSE(decoded_secret_images, target_secret_images.detach())
            g_loss_on_decoder2 = model.criterion_MSE(decoded_messages, messages.clone().detach())

            #target_clean_images = torch.zeros_like(secret_images)

            #g_loss_on_decoder3 = model.criterion_MSE(decoded_clean_images, target_clean_images.detach())


            # full loss
            g_loss = g_loss_on_encoder + g_loss_on_discriminator +\
                     g_loss_on_encoder_perceptual + \
                     g_loss_on_decoder1 + \
                     g_loss_on_decoder2 * (g_loss_on_decoder2.item() / g_loss_on_encoder.item()) #+ g_loss_on_decoder3

            g_loss.backward()
            model.opt_encoder_decoder.step()

            '''
            decoded message error rate /Dual
            '''
            error_rate = model.decoded_message_error_rate_batch(decoded_messages.clone().detach(), messages.clone().detach())
            psnr = -kornia.losses.psnr_loss((encoded_images.clone().detach() + 1) / 2, (raw_images.clone().detach() + 1) / 2, 1)
            ssim = 1 - 2 * kornia.losses.ssim_loss((encoded_images.clone().detach() + 1) / 2, (raw_images.clone().detach() + 1) / 2, window_size=11, reduction="mean")

            total_error_rate.update(error_rate, 1)
            total_g_loss_on_encoder.update(g_loss_on_encoder.item(), 1)
            total_g_loss_on_encoder_perceptual.update(g_loss_on_encoder_perceptual.item(), 1)
            total_g_loss_on_decoder1.update(g_loss_on_decoder1.item(), 1)
            total_g_loss_on_decoder2.update(g_loss_on_decoder2.item(), 1)
            #total_g_loss_on_decoder3.update(g_loss_on_decoder3.item(), 1)
            total_g_loss_on_d_loss.update(d_loss.item(), 1)
            total_g_loss_on_discriminator.update(g_loss_on_discriminator.item(), 1)
            total_PSNR.update(psnr.item(), 1)
            total_SSIM.update(ssim.item(), 1)

        result = {
            "g_loss": g_loss,
            "error_rate": total_error_rate.avg,
            "g_loss_on_encoder": total_g_loss_on_encoder.avg,
            "g_loss_on_encoder_perceptual": total_g_loss_on_encoder_perceptual.avg,
            "g_loss_on_decoder1": total_g_loss_on_decoder1.avg,
            "g_loss_on_decoder2": total_g_loss_on_decoder2.avg,
            #"g_loss_on_decoder3": total_g_loss_on_decoder3.avg,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "d_loss": d_loss.item(),
            "PSNR": total_PSNR.avg,
            "SSIM": total_SSIM.avg
        }
    return result