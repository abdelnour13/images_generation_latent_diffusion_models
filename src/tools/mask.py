import sys
import os
sys.path.append('../..')
import definitions as D
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from src.datasets import ImageDirectory
from argparse import ArgumentParser
from dataclasses import dataclass

@dataclass
class Args:
    dataset : str
    threshold : int

def main(args : Args):
    
    ### *** Load dataset *** ###
    dataset = ImageDirectory(
        dataset = args.dataset,
        type='image',
        return_metadata=False
    )

    ### *** Create mask directory *** ###
    mask_dir = os.path.join(D.DATASETS[args.dataset].root, 'masks')
    os.makedirs(mask_dir, exist_ok=True)

    ### *** Create masks *** ###
    for data in tqdm(dataset):

        # Generate mask
        idx = data['id']
        img = np.array(data['data'])
        mask = img[:,:,0] > args.threshold

        # Save mask
        image_id = dataset.splits_df.iloc[idx].image_id
        image_name = os.path.splitext(image_id)[0]
        mask_path = os.path.join(mask_dir, image_name + '.png')
        mask = Image.fromarray(mask.astype(np.uint8)*255)
        mask.save(mask_path)

if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--threshold', type=int, required=True, help='Threshold value')

    args = parser.parse_args()
    args = Args(**vars(args))

    main(args)