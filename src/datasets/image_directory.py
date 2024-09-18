import pandas as pd
import os
import sys
import numpy as np
sys.path.append('../..')
from torch.utils.data import Dataset
from definitions import DATASETS
from typing import Optional,Callable,Literal
from PIL import Image

class ImageDirectory(Dataset):

    def __init__(self,
        dataset : str,
        split : Optional[str] = None,
        transform : Optional[Callable] = None,
        type : Literal['image','latent'] = 'image',
        return_metadata : bool = False
    ) -> None:
        
        super().__init__()

        self.dataset = DATASETS[dataset]
        self.split = split
        self.transform = transform
        self.type = type
        self.latents_dir = os.path.join(self.dataset.root, 'features')
        self.return_metadata = return_metadata

        if self.return_metadata:

            if self.dataset.metadata_file is None:
                raise ValueError("Metadata file not found")
            else:
                self.metadata = pd.read_csv(self.dataset.metadata_file)

        self._verify()

        self.splits_df = pd.read_csv(self.dataset.splits_file)

        if split is not None:
            splits = ['train', 'val', 'test']
            self.splits_df = self.splits_df[self.splits_df['partition'] == splits.index(split)]

    def _verify(self) -> None:
        
        if self.split is not None and self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be one of ['train', 'val', 'test']")
        
        if self.type == 'latent' and not os.path.exists(self.latents_dir):
            raise FileNotFoundError(f"Latents directory not found at {self.latents_dir}")
        
    def __len__(self) -> int:
        return len(self.splits_df)
    
    def __getitem__(self, idx : int) -> dict:

        row = self.splits_df.iloc[idx]

        image_id = row['image_id']

        if self.type == 'image':
            image_path = os.path.join(self.dataset.images_dir, image_id)
            image = Image.open(image_path).convert('RGB')
        else:
            latent_path = os.path.join(self.latents_dir, f'{os.path.splitext(image_id)[0]}.npy')
            image = np.load(latent_path)

        if self.transform is not None:
            image = self.transform(image)

        data = {
            'id' : idx,
            'data' : image,
        }

        if self.metadata is not None:
            metadata = self.metadata[self.metadata['image_id'] == image_id].iloc[0]
            data['metadata'] = metadata.drop('image_id').to_dict()

        return data