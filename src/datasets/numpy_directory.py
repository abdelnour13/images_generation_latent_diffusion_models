import os
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Callable, Optional

class NumpyDirectory(Dataset):

    def __init__(self, 
        root : str, 
        transform : Optional[Callable] = None,
        split : Optional[str] = None,
    ) -> None:
        super(NumpyDirectory, self).__init__()

        self.root = root
        self.transform = transform
        self.split = split
        self.files = self._find_files()

    def _find_files(self) -> list[str]:

        files = []
        splits = os.listdir(self.root)

        if self.split is not None:

            if self.split not in splits:
                raise ValueError(f"Split {self.split} not found in {self.root}")
            
            splits = [self.split]
            
        for split in splits:
            path = os.path.join(self.root, split)
            files.extend([os.path.join(split, file) for file in os.listdir(path)])

        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx : int) -> Any:

        file = self.files[idx]
        path = os.path.join(self.root, file)
        data = np.load(path)

        if self.transform is not None:
            print('Transforming data')
            data = self.transform(data)

        return {
            'latent' : data,
            'idx' : idx,
        }