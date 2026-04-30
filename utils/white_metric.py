import os
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

def load_image(path):
    """Load image and normalize to [0,1] float32"""
    img = Image.open(path).convert('RGB')
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def compute_s_white(dec_path, eps=1e-8, resolution=256):
    """
    Compute S_white for all images with matching filenames.
    Returns a dict: {filename: S_white}
    """
    if resolution == 128:
        I_s = load_image(r'E:\PythonProject\Dualmark\data\secret_image\BlackWhite128.jpg')
    else:
        I_s = load_image(r'E:\PythonProject\Dualmark\data\secret_image\BlackWhite256.jpg')

    I_w = np.ones_like(I_s, dtype=np.float32)

    I_hat = load_image(dec_path)

    d_s = mse(I_hat, I_s)
    d_w = mse(I_hat, I_w)

    S_white = d_s / (d_s + d_w + eps)

    return S_white

def calculate_ssim(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


    _, binary1 = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, 128, 255, cv2.THRESH_BINARY)

    return ssim(binary1, binary2)


def process_images(real_folder, fake_folder, threshold, output_file):

    white_values = []
    predicted_labels = []
    ground_truth_labels = []

    with open(output_file, "w") as f:
        f.write("Filename\t\tWhiteScore\t\tLabel\n")


        for filename in tqdm(os.listdir(real_folder)):
            file_path = os.path.join(real_folder, filename)
            score = compute_s_white(file_path)
            white_values.append(score)
            predicted_labels.append(1 if score >= threshold else 0)
            ground_truth_labels.append(0)
            f.write(f"{'EI_' + filename}\t\t{score:.4f}\t\tREAL\n")


        for filename in tqdm(os.listdir(fake_folder)):
            file_path = os.path.join(fake_folder, filename)
            score = compute_s_white(file_path)
            white_values.append(score)
            predicted_labels.append(1 if score >= threshold else 0)
            ground_truth_labels.append(1)
            f.write(f"{'NI_' +filename}\t\t{score:.4f}\t\tFAKE\n")

        if len(set(ground_truth_labels)) > 1:
            acc = accuracy_score(ground_truth_labels, predicted_labels)
            auc = roc_auc_score(ground_truth_labels, white_values)
            print(f"Accuracy (ACC): {acc:.4f}")
            print(f"AUC: {auc:.4f}")
            f.write(f"Accuracy\t\t{acc:.4f}\t\tAUC\t\t{auc:.4f}\n")

    print(f"Total images processed: {len(white_values)}")
    print(f"Results saved to {output_file}")

    return white_values

method = 'UniFaceReen'
dataset = 'FFHQ'
output_txt = 'white_record\\' + dataset + '\\256\\' + method + '.txt'

real_folder = r'E:\PythonProject\Dualmark\results\\' + dataset + '\\256\\common\\Identity\SI'
fake_folder = r'E:\PythonProject\Dualmark\\results\\' + dataset + '\\256\\deepfake\\' + method + '\SI'

threshold_value = 0.5   

process_images(real_folder, fake_folder, threshold_value, output_txt)