import torch
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
from PIL import Image
from typing import Optional
from imagesize import get

def seed_everything(seed : int) -> None:

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
    
def display_image_in_actual_size(
    image : np.ndarray,
    ax : Optional[plt.Axes] = None,
) -> plt.Axes:

    dpi = 80
    height, width, depth = image.shape
    ax = ax or plt.gca()

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image)

    return ax