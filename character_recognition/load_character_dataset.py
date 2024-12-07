import os
import torch
from torch.utils.data import Dataset, DataLoader
import string
import numpy as np


class TraceDataset(Dataset):
    def __init__(self, root_dir=None, preloaded_data=None, normalize=False):
        """
            root_dir (str): directory with all the data files
            preloaded_data (tuple): A tuple containing preloaded data and labels
            normalize (bool): Flag to always normalize the data
        """
        self.normalize = normalize
        
        if preloaded_data is not None:
            self.data, self.labels = preloaded_data
        else:
            self.root_dir = root_dir
            self.data = []
            self.labels = []
            self.label_map = {str(i): i for i in range(10)}  # Map 0-9
            self.label_map.update({c: i + 10 for i, c in enumerate(string.ascii_lowercase)})  # Map a-z
            self._load_data()

            if self.normalize:
                self.mean, self.std = self._compute_mean_std()

    def _load_data(self):
        """
        loads data from provided directory.
        """
        for label_folder in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label_folder)
            if os.path.isdir(label_path):
                for trace_file in os.listdir(label_path):
                    if trace_file.startswith("trace") and trace_file.endswith(".txt"):
                        file_path = os.path.join(label_path, trace_file)
                        print(file_path) 
                        with open(file_path, 'r') as f:
                            data_points_str = f.read().split('\n')[:-1]
                            data_points = []
                            for line in data_points_str:
                                coordinates = line.split(',')
                                x, y = float(coordinates[0]), float(coordinates[1])
                                data_points.append([x, y])  
                            self.data.append(data_points)
                            self.labels.append(self.label_map[label_folder])

    def _compute_mean_std(self):
       
        all_data = np.array([point for trace in self.data for point in trace])  # Shape: (num_samples, 2)
        mean = np.mean(all_data, axis=0)  
        std = np.std(all_data, axis=0)    
        
        print(f"Mean: {mean}, Std: {std}")
        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        x = np.array(x)
        if self.normalize:
            x = (x - self.mean) / self.std
        x = torch.tensor(x, dtype=torch.float32)
        return x, torch.tensor(y, dtype=torch.long)



dataset = TraceDataset(root_dir="character_recognition/HCMYO-A/data/S01", normalize=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

torch.save((dataset.data, dataset.labels), 'character_recognition/dataset_data_labels.pth')
print(f"Dataset Size: {len(dataset.data)}")
print(f"Dataset Labels Size: {len(dataset.labels)}")

dataloader_config = {
    'batch_size': 32,
    'shuffle': True,
}
torch.save(dataloader_config, 'character_recognition/dataloader_config.pth')
#torch.save((dataset.mean, dataset.std), 'normalization_params.pth')
print("configuration saved.")
