# WavMark
We provide a sample code of the WavMark method for defending against StarGAN.

## Usage

### Installation

1. Prepare the Environment
   Install the lib by pip (recommend)
    ```
    pip install -r requirements.txt
    ```
2. Prepare the Datasets

   Download the [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ) datasets
4. Prepare the Model Weights
   
   For your convenient usage, we prepare the weights download link in [Google Drive](https://drive.google.com/file/d/1TQUFE9tycLaxMGVW4cnV_0FvaZ3lTjxN/view?usp=sharing).


The name of stargan generator model weights is *200000-G.ckpt*. We also prepare a WavMark model for you to test its performance named *256.tar*. Due to the file upload size limitation, we will soon make the pre-trained model publicly available online.

### Inference

   ```python
   python test.py
   ```
