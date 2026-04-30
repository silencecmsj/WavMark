import os
import numpy as np
from PIL import Image

def load_image(path):
    """Load image and normalize to [0,1] float32"""
    img = Image.open(path).convert('RGB')
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def compute_s_white(decoded_dir, original_dir, eps=1e-8):
    """
    Compute S_white for all images with matching filenames.
    Returns a dict: {filename: S_white}
    """
    results1 = {}
    results2 = {}
    results3 = {}

    filenames = sorted(os.listdir(decoded_dir))

    ori_path = os.path.join(original_dir)
    I_s = load_image(ori_path)

    I_w = np.ones_like(I_s, dtype=np.float32)

    for name in filenames:
        dec_path = os.path.join(decoded_dir, name)

        if not (os.path.exists(ori_path)):
            print(f"[Warning] Missing counterpart for {name}, skip.")
            continue

        I_hat = load_image(dec_path)

        d_s = mse(I_hat, I_s)
        d_w = mse(I_hat, I_w)

        S_white = d_s / (d_s + d_w + eps)
        results1[name] = S_white
        results2[name] = d_s
        results3[name] = d_w

    return results1, results2, results3

decoded_dir  = r"E:\PythonProject\Dualmark\results\CelebAHQ\256\common\GaussianNoise\SI"
original_dir = r"E:\PythonProject\Dualmark\data\secret_image\BlackWhite256.jpg"

s_white_dict, s_ds_dict, s_dw_dict = compute_s_white(decoded_dir, original_dir)

avg_s_white = np.mean(list(s_white_dict.values()))
avg_ds = np.mean(list(s_ds_dict.values()))
avg_dw = np.mean(list(s_dw_dict.values()))
#std_s_white = np.std(list(s_white_dict.values()))

print('S_white:', avg_s_white, 'd_s:', avg_ds, 'd_w:', avg_dw)