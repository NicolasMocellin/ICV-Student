'''
1️⃣ Install PyTorch separately
CPU VERSION:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

GPU/Cuda VERSION (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

to verify installation:
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())

to uninstall (if needed):
pip uninstall torch torchvision -y

2️⃣ Install the rest
pip install -r requirements.txt

3️⃣ Apply the Basicsr patch (only once)

in .venv\Lib\site-packages\basicsr\data\degradations.py
replace
from torchvision.transforms.functional_tensor import rgb_to_grayscale
with
from torchvision.transforms.functional import rgb_to_grayscale

'''

import torch
import numpy as np
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from utils import show
import os
from tqdm import tqdm
from P0_Model_Tensorflow import create_model_and_evaluate

def enhance_image_with_esrgan(img, upsampler, out_scale=1):
    """Apply Real-ESRGAN enhancement to a single image.

    Real-ESRGAN uses a Generative Adversarial Network (GAN) trained on paired
    low/high-quality images to recover fine details and textures.

    Args:
        img: Input image
        upsampler: RealESRGANer instance (initialized model)
        out_scale: Upscaling factor (1 = no resolution change, only quality enhancement)

    Returns:
        Enhanced image
    """
    try:
        output, _ = upsampler.enhance(img, outscale=out_scale)
        return output
    except Exception as e:
        print(f"Processing error : {e}")

def init_esrgan():
    '''
    Initialize the Real-ESRGAN model for image enhancement.
    Returns:
        upsampler: Initialized RealESRGANer instance
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Initialisation of Real-ESRGAN on {device}...")
   
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # weights
    model_path = r'weights\RealESRGAN_x4plus.pth'

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=(device == 'cuda'),
        device=device,
    )

    return upsampler



if __name__ == "__main__":

    '''
    # example of usage
    upsampler = init_esrgan()
    img = cv2.imread(r'./images/esrgan.jpeg', cv2.IMREAD_UNCHANGED)
    enhanced = enhance_image_with_esrgan(img, upsampler, out_scale=1)
    '''

    # 1- Test ESRGAN on esrgan.jpeg (x1 and x4 upscale)
	
    # 2- Test ESRGAN on 2under-exposed.jpeg
	
    # 3- Improve equalized (CLAHE) dataset and test set images (x1 upscale) and test the model

    # WRITE ALL REMARKS AND CONCLUSIONS IN THIS FILE AS COMMENTS
