import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.signals import (
    generate_cos_signal,
    generate_pulse_signal,
    generate_sin_signal,
)


class EEG_Dataset(Dataset):
    
    def __init__(self, root):
        super(EEG_Dataset, self).__init__()
        self.root = Path(root)
        self.pkl_names = []
        for i in self.root.rglob('*.pkl'):
            self.pkl_names.append(i)

    def __len__(self):
        return len(self.pkl_names) * 720
    
    def __getitem__(self, index):
        i = index // 720
        pkl_name = self.pkl_names[i]
        patien_id = pkl_name.name.split('/')[-1].split('-')[0]
        with open(pkl_name, 'rb') as f:
            data = pickle.load(f)
        x = data.to_numpy()[index % 720, :, np.newaxis].reshape(1, -1)
        return patien_id, x

    def get_sample_shape(self):
        raise NotImplementedError

class ESR(Dataset):
    
    def __init__(self, root):
        super(ESR, self).__init__()
        path = Path(root) / 'Epileptic_Seizure_Recognition/Epileptic_Seizure_Recognition.csv'
        data = pd.read_csv(path)
        data = data.to_numpy()[:, 1:]
        self.X = data[:, :-1]
        self.Y = data[:, -1]
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, index):
        
        # convert x type to float32
        x = self.X[index].astype(np.float32)
        
        # add some 0 at the end of vector and convert it to 14 x 14
        x = np.pad(x, (0, 18), 'constant', constant_values=(0))
        x = x.reshape(14, -1)
        
        # # use 0 to padding to 28 x 28
        # x = np.pad(x, ((7, 7), (7, 7)), 'constant', constant_values=0)
        
        # TODO 尝试不同的normalization方法（均值方差）
        # normalization
        x_max = x.max()
        x_min = x.min()
        x = (x - x_min) / (x_max - x_min)
        
        # convert its shape to 1 x 28 x 28
        x = np.expand_dims(x, axis=0)
        
        # get y
        y = self.Y[index]
        
        return x, y

class Single(Dataset):
    def __init__(self):
        
        amplitude: float = 1
        frequency: float = 0.5
        phase: float = 0
        duration: float = 5
        sampling_rate: float = 178 / duration

        self.X = None
        
        for _ in range(1000):
            _, signal = generate_sin_signal(amplitude, frequency, np.random.random()*10, duration, sampling_rate, noise=True)
            if self.X is None:
                self.X = signal
            else:
                self.X = np.vstack([self.X, signal])
        
        self.X = np.expand_dims(self.X, axis=1)
        
        x_max: float = self.X.max()
        x_min: float = self.X.min()
        
        self.X = (self.X - x_min) / (x_max - x_min)
        
    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> np.ndarray:        
        return self.X[index].astype(np.float32)