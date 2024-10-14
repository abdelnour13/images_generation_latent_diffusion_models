import torch
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from torch import Tensor
from tqdm.auto import tqdm
from PIL import Image
from typing import Optional,Any
from imagesize import get
from collections import OrderedDict
from torch.nn import functional as F
from my_palette import PaletteCreation

def move_data_to_device(data : Any, device : str | torch.device) -> Any:

    if isinstance(data, Tensor):
        return data.to(device)
    
    if isinstance(data, dict):
        return {key: move_data_to_device(value, device) for key, value in data.items()}
    
    if isinstance(data, list):
        return [move_data_to_device(value, device) for value in data]
    
    if isinstance(data, tuple):
        return tuple(move_data_to_device(value, device) for value in data)
    
    raise ValueError(f"Data type {type(data)} not supported")

def hex_to_rgb(hex : str) -> tuple[int, int, int]:

    hex = hex.lstrip('#')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

    return rgb

def get_palette(
    image : Image.Image,
    n_colors : int = 5
) -> np.ndarray:
    
    palette = PaletteCreation()
    palette = palette.get_palette(image, n_colors) # list[str] 
    palette = [hex_to_rgb(color) for color in palette] # list[tuple[int, int, int]]
    palette = np.array(palette) # (n_colors, 3)

    return palette

def get_last_checkpoint(checkpoints_dir : str,device : str | torch.device | None = None) -> Optional[OrderedDict]:

    checkpoints = os.listdir(checkpoints_dir)
    checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.endswith('.pt') or checkpoint.endswith('.pth')]
    checkpoints = sorted(checkpoints)

    if len(checkpoints) == 0:
        return None

    last_checkpoint = checkpoints[-1]
    last_checkpoint = os.path.join(checkpoints_dir, last_checkpoint)

    return torch.load(last_checkpoint,map_location=device)

def seed_everything(seed : int,use_additional : bool = False) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_metadata(root : str,filename : Optional[str] = None) -> pd.DataFrame:

    if filename is not None and os.path.exists(filename):
        return pd.read_csv(filename)

    names = os.listdir(root)

    metadata = {
        'name': [],
        'height': [],
        'width': [],
        'aspect_ratio': [],
        'size': [],
        'resolution': []
    }

    for name in tqdm(names):

        path = os.path.join(root, name)
        width, height = get(path)

        metadata['name'].append(name)
        metadata['height'].append(height)
        metadata['width'].append(width)
        metadata['aspect_ratio'].append(width / height)
        metadata['size'].append(os.path.getsize(path))
        metadata['resolution'].append(f'{width}x{height}')

    metadata = pd.DataFrame(metadata)

    if filename is not None:
        metadata.to_csv(filename, index=False)

    return metadata

def display_images(
    images : list[Image.Image],
    nrows : int,
    ncols : int,
) -> None:
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i, (image, ax) in enumerate(zip(images, axes)):
        ax.imshow(image)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def diaply_random_images(
    root : str,
    nrows : int,
    ncols : int,
) -> None:

    names = os.listdir(root)
    selected_names = random.sample(names, nrows * ncols)
    images = [Image.open(os.path.join(root, name)) for name in selected_names]

    display_images(images, nrows, ncols)

def load_json(path : str) -> dict:

    with open(path, 'r') as file:
        return json.load(file)
    
def save_json(data : dict, path : str, indent : Optional[int] = 4) -> None:
    
    with open(path, 'w') as file:
        json.dump(data, file, indent=indent)

def make_grid(
    images : Tensor,
    h : int,
    w : int,
    gap : int = 4,
    gap_value : int | float = 255,
) -> Tensor:
    
    B,C,H,W = images.shape[-4:]
    rest = images.shape[:-4]
    S = len(rest)
    
    if w * h < B:
        images = images[:w*h,]
    elif w * h > B:
        padding = torch.ones(*rest,w*h - B,C,H,W) * gap_value
        images = torch.cat([images,padding],dim=-4)
        B = w * h


    images = images.reshape(np.prod(rest).astype(int) * B,C,H,W) # (B,C,H,W)
    images = F.pad(images,(gap,gap,gap,gap),value=gap_value)  # (B,C,H,W)
    images = images.reshape(*rest,B,*images.shape[1:]) # (...,B,C,H,W)
    images = images.permute(*range(S),S,S+2,S+3,S+1) # (...,B,H,W,C)
    images = images.reshape(*rest,h,w,*images.shape[-3:]) # (...,G_h,G_w,H,W,C)
    images = images.permute(*range(S),S,S+2,S+1,S+3,S+4) # (...,G_h,H,G_w,W,C)
    images = images.flatten(S,S+1).flatten(S+1,S+2) # (...,H,W,C)

    return images