import sys
import os
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path

epu_path = Path(__file__).resolve().parent
sys.path.append(str(epu_path))

import torch
import numpy as np

from PIL import Image
from glob import glob
from utils.epu_utils import load_model, preprocess_image, EPUConfig


def user_arguments() -> argparse.Namespace:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate EPU model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing the saved model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to be processed')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for the model')
    args = parser.parse_args()
    return args


def main():

    args = user_arguments()

    # Configuration
    epu_config_path = os.path.join(os.getcwd(), *args.model_path.split("/"), "epu.config")
    train_config_path = os.path.join(os.getcwd(), *args.model_path.split("/"), "train.config")
    
    epu_config = EPUConfig.load_config_object(epu_config_path)
    train_parameters = EPUConfig.load_config_object(train_config_path)

    # Load model
    print(f"\n[+] Loading model from {args.model_path}...")
    epu_config_path = os.path.join(os.getcwd(), *args.model_path.split("/"), "epu.config")
    checkpoint_path = os.path.join(os.getcwd(), *args.model_path.split("/"), f"{epu_config.experiment_name}.pt")
    
    model = load_model(checkpoint_path, 
                       epu_config_path, 
                       mode=train_parameters.mode, 
                       label_mapping=train_parameters.label_mapping.__dict__, 
                       confidence=args.confidence)
    model.to("cuda")

    image_path = args.image_path
    print(f"[+] Processing {image_path}...\n")
    # Process image
    image = preprocess_image(image_path, train_parameters.input_size)
    output = model(torch.tensor(image).unsqueeze(1).to("cuda")).detach().cpu().numpy()
    
    if train_parameters.mode == "binary":
        output = 1 if output > args.confidence else 0
    elif train_parameters.mode == "multiclass":
        output = np.argmax(output)
    
    labels = dict((v,k) for k,v in train_parameters.label_mapping.__dict__.items())
    output = labels[output]
    print(f"[+] Output: {output}")
    
    # TODO: Add interpretation support for multiclass problems
    model.plot_rss(savefig=True, input_image_name=os.path.basename(image_path))
    model.get_prm(savefig=True, 
                    refine_prm=True, 
                    height=train_parameters.input_size, 
                    width=train_parameters.input_size, 
                    input_image=Image.open(image_path),
                    input_image_name=os.path.basename(image_path))
    
    print(f"[+] Interpretations saved to interpretations/{epu_config.experiment_name}")
        
if __name__ == "__main__":
    main() 