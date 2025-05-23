import numpy as np
import torch
from torch.utils.data import Dataset
import os


class NumpyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Get list of .npy file paths
        self.file_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".npy")]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Load the .npy file as a NumPy array
        image = np.load(file_path).astype(np.float32)  # Ensure it's float32 for PyTorch compatibility
        
        # If grayscale but missing channel dimension, add (C, H, W) format
        if image.ndim == 2:  
            image = np.expand_dims(image, axis=0)  # Convert (H, W) → (1, H, W)
        
        # Convert NumPy array to PyTorch tensor
        image = torch.from_numpy(image)
        # normalize to [0,1]
        image = image / torch.max(image)


        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        return image



class SwAVDataset(Dataset):
    def __init__(self,imgs_path,patch_size=11, patch_stride=8,
                        low_info_thresh=None,adjust_scale=False,filter_data=None):
        
        self.data =  np.load(imgs_path)

        self.max_value= np.nanmax(self.data) 
        self.corrected_mean = np.nanmean(self.data) /self.max_value
        self.corrected_std = np.nanstd(self.data) /self.max_value

        self.data = torch.tensor(self.data, dtype=torch.float64)

        if filter_data:
            random_dx= np.random.randint(0, self.data.shape[0],size=filter_data)    
            self.data= self.data[random_dx]

        self.imgs = self._extract_patches(self.data, patch_size, patch_stride)  
        
        if adjust_scale: 
            self.imgs*=1000

        if low_info_thresh is not None:
            print("######### Initial datasetsize:",self.imgs.shape[0],"\n")
            self.imgs = self._remove_low_info_data(self.imgs ,low_info_thresh)
            print("Final datasetsize:",self.imgs.shape[0]," #########\n")

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.Tensor(self.imgs[idx]).to(torch.uint8)

    def __len__(self):
        return len(self.imgs)

    def _extract_patches(self, data, patch_size, patch_stride,exclude_nans=True):
        """
        Extrai patches usando unfold para divisões com ou sem sobreposição.
        """
        # b, h, w = data.shape
        patches = data.unfold(1, patch_size, patch_stride).unfold(2, patch_size, patch_stride)
        patches = patches.contiguous().view(-1, patch_size, patch_size)

        if exclude_nans:
            patches = patches[~torch.isnan(patches).any(dim=(1, 2))] 

        return patches
    
    def _remove_low_info_data(self, data, quantile_thresh=0.2):

        patch_variance = data.var(dim=(1, 2))
        variance_threshold = patch_variance.quantile(quantile_thresh)

        low_info = (patch_variance <= variance_threshold) 
        informative_indices = torch.nonzero(~low_info).squeeze()

        return data[informative_indices]
