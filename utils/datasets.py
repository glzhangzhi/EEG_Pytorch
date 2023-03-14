import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


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
        x = self.X[index].astype(np.float32)
        x = np.pad(x, (0, 18), 'constant', constant_values=(0))
        x = x.reshape(14, -1)
        x = np.pad(x, ((7, 7), (7, 7)), 'constant', constant_values=0)
        x_max = x.max()
        x_min = x.min()
        x = (x - x_min) / (x_max - x_min)
        x = np.expand_dims(x, axis=0)
        y = self.Y[index]
        return x, y
    
    def get_x_shape(self):
        return (1, 1, 178)
    
    def get_y_shape(self):
        return (1)