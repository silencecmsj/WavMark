import argparse

parser = argparse.ArgumentParser(description='Defense')

# config para
parser.add_argument('--csv_file_path', default=r'data\CelebA_256.csv', help='features csv file path')
parser.add_argument('--checkpoint_path', type=str,
    # 256
    default=r'checkpoints', help='checkpoint path')
parser.add_argument('--batch_size', type=int, default=32, help='number of instances in a batch of data (default: 64)')
parser.add_argument('--status', type=int, default=0, help='train:0, test:1')
parser.add_argument('--message_length', type=int, default=30, help='message length')
parser.add_argument('--resolution', type=str, default='128', help='image size')
parser.add_argument('--secret_image_path', type=str,
                    default=r'data\secret_image\BlackWhite128.jpg', help='path')
parser.add_argument('--face_coordinates', type=str,
                    default=r'data\face_coordinates_128.json')
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
                                                                          #'StarGAN(args)',
                                                                           #'HiSD(args)',
                                                                          #'SimSwap(args)',
                                                                          #'InfoSwap(args)',
                                                                          #'UniFaceSwap(args)',
                                                                          #'UniFaceReen(args)',
                                                                           'GANimation(args)',
                                                                          #'HyperReenact(args)',
                                                                          #'DiffAE(args)',
                                                                          #'DiffusionClip(args)'
                                                                    ], help='weight')


parser.add_argument('--dataset', default="CelebAHQ", type=str, help='dataset Name')
parser.add_argument('--epoch', type=int, default=300, help='maximum iteration number (default: 100)')
parser.add_argument('--gpus', default="cuda:0", type=str, help='gpus')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
#parser.add_argument('--weight', type=float, default=0.2, help='weight_decay (default: 0.2)')
parser.add_argument('--workers', type=int, default=0, help='number of workers in dataloader')
parser.add_argument('--attention_encoder', type=str, default='se', help='attention layer')
parser.add_argument('--attention_decoder', type=str, default='se', help='attention layer')
parser.add_argument('--beta', type=float, default=0.5, help='message length')
parser.add_argument('--weight', type=list, default=[1, 1, 1], help='weight')
parser.add_argument('--message_range', type=list, default=[0, 1], help='weight')