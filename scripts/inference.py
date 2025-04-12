import sys
sys.path.append("path/to/epu-cnn-torch")
import os
import torch

from PIL import Image
from glob import glob
from typing import List
from utils.epu_utils import load_model, preprocess_image
from model.epu import EPU

def main():
    # Configuration
    model_path = "checkpoints/epu_banapple_250epochs.pt"
    config_path = "configs/binary_epu_config.yaml"
    test_dir = "path/to/test/dir"
    input_size = 128
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, config_path)
    
    # Get test images
    test_images = glob(os.path.join(test_dir, "*"))
    
    # Process each test image
    for image_path in test_images:
        print(f"\nProcessing {image_path}...")
        print(image_path)
        # Process image
        image = preprocess_image(image_path, input_size)
        print(model(torch.tensor(image).unsqueeze(1)))
        model.plot_rss(savefig=True)
        model.get_prm(savefig=True, 
                      refine_prm=True, 
                      height=input_size, 
                      width=input_size, 
                      input_image=Image.open(image_path))
        
if __name__ == "__main__":
    main() 