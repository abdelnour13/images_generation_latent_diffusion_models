import pandas as pd
import os
import sys
import numpy as np
sys.path.append('../..')
from torch.utils.data import Dataset
from definitions import DATASETS
from typing import Optional,Callable
from PIL import Image
from src.utils import get_palette

class ImageDirectory(Dataset):

    def __init__(self,
        dataset : str,
        split : Optional[str] = None,
        transform : Optional[Callable] = None,
        color_palette_size : int = 5
    ) -> None:
        
        super().__init__()

        self.dataset = dataset
        self.split = split
        self.transform = transform
        self.color_palette_size = color_palette_size

        self._verify()

        self.splits_df = pd.read_csv(DATASETS[dataset].splits_file)

        if split is not None:
            splits = ['train', 'val', 'test']
            self.splits_df = self.splits_df[self.splits_df['partition'] == splits.index(split)]

    def _verify(self) -> None:

        if self.dataset not in DATASETS:
            raise ValueError(f"Dataset {self.dataset} not available")
        
        if self.split is not None and self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be one of ['train', 'val', 'test']")
        
    def __len__(self) -> int:
        return len(self.splits_df)
    
    def __getitem__(self, idx : int) -> dict:

        row = self.splits_df.iloc[idx]

        image_id = row['image_id']
        image_path = os.path.join(DATASETS[self.dataset].images_dir, image_id)

        image = Image.open(image_path).convert('RGB')
        # palette = get_palette(np.array(image).reshape(-1,3), self.color_palette_size) / 255.0

        if self.transform is not None:
            image = self.transform(image)

        return {
            'id' : idx,
            'image' : image,
        }