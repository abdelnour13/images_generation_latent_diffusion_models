import sys
import os
sys.path.append('../..')
import definitions as D
import numpy as np
from tqdm.auto import tqdm
from src.datasets import ImageDirectory
from argparse import ArgumentParser
from dataclasses import dataclass
from src.utils import get_palette,save_json

@dataclass
class Args:
    dataset : str
    resolution : int
    n : int

def main(args : Args):
    
    ### *** Load dataset *** ###
    dataset = ImageDirectory(
        dataset = args.dataset,
        type='image',
        return_metadata=False
    )

    palettes = {}
    palettes_path = os.path.join(D.DATASETS[args.dataset].root, 'palettes.json')

    ### *** Get color palette *** ###
    for data in tqdm(dataset):

        # Generate color palette
        data['data'] = data['data'].resize((args.resolution, args.resolution))
        img = np.array(data['data'])
        img = img.reshape(-1, 3)

        # Get color palette
        palette = get_palette(img, args.n).tolist()

        # Save color palette
        image_id = dataset.splits_df.iloc[data['id']].image_id
        palettes[image_id] = palette

    save_json(palettes, palettes_path, None)

if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--resolution', type=int, required=True, help='Resolution')
    parser.add_argument('--n', type=int, required=True, help='Number of colors')

    args = parser.parse_args()
    args = Args(**vars(args))

    main(args)