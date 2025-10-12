import lpips
import torch
import torchvision
from torch import nn
from torch.nn.functional import mse_loss
from torchvision.models import VGG19_Weights

from module.EncoderDecoder import EncoderDecoder
from module.Patch_Discriminator import Patch_Discriminator


class DualMark:

    def __init__(self, args,
                 lr, beta1, weight):
        # device
        self.device = args.device
        # loss function
        self.criterion_MSE = nn.MSELoss().to(args.device)
        #self.lpips_loss = PerceptualLoss().to(args.device)
        self.lpips_loss = lpips.LPIPS(net='vgg').to(args.device)
        #self.criterion_LPIPS = lpips.LPIPS().to(args.device)

        # network
        self.encoder_decoder = EncoderDecoder(args).to(args.device)
        self.discriminator = Patch_Discriminator().to(args.device)

        # self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        # self.discriminator = torch.nn.DataParallel(self.discriminator)


        # mark "cover" as 1, "encoded" as 0
        self.label_cover = 1
        self.label_encoded = 0

        for p in self.encoder_decoder.noise.parameters():
            p.requires_grad = False

        # optimizer
        self.opt_encoder_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr, betas=(beta1, 0.999))

        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[0]

        message = message.gt(0.5)
        decoded_message = decoded_message.gt(0.5)
        error_rate = float(sum(message != decoded_message)) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages):
        error_rate = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
        error_rate /= batch_size
        return error_rate

    def save_model(self, path_encoder_decoder: str, path_discriminator: str):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator.module.state_dict(), path_discriminator)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder), strict=False)

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator))
