
from PIL.Image import Image
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import nn, Tensor

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class CopyTransform(nn.Module):

    def __init__(self, n_copies=5):
        super().__init__()

        self.n_copies = n_copies
        self.copy = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])

    def forward(self, img: Tensor):
        width, height = img.shape
        half_width, half_height = int(width // 2), int(height // 2)

        original_img = img.clone()
        for _ in range(self.n_copies):
            img_copy = self.copy(original_img)
            empty_pixels = torch.all(img == 0, dim=0)
            if not torch.any(empty_pixels):
                return img
            
            # Random generate center
            indexes = torch.where(empty_pixels)
            x_ind = torch.randint(0, indexes[0].shape[0], (1,))
            y_ind = torch.randint(0, indexes[1].shape[0], (1,))
            x, y = indexes[0][x_ind], indexes[1][y_ind]
            
            # Find original image boundaries
            x0, y0 = max(0, x - half_width), max(0, y - half_height)
            x1, y1 = min(width, x + half_width), min(height, y + half_height)
            
            # Find copy boundaries
            dx, dy = x1 - x0, y1 - y0
            img_copy = img_copy[
                :,
                half_width - dx//2:half_width + dx//2 + dx%2,
                half_height - dy//2:half_height + dy//2 + dy%2
            ]

            img[:, x0:x1, y0:y1] = torch.where(
                empty_pixels[x0:x1, y0:y1],
                img_copy,
                img[:, x0:x1, y0:y1]
            )

        return img

class MultiViewTransform:
    def __init__(self, transforms: Sequence[T.Compose]):
        self.transforms = transforms

    def __call__(self, image: Union[Tensor, Image]) -> Union[List[Tensor], List[Image]]:
        return [transform(image) for transform in self.transforms]    

class MultiCropTranform(MultiViewTransform):
    def __init__(
        self,
        crop_sizes: Tuple[int, ...],
        crop_counts: Tuple[int, ...],
        crop_min_scales: Tuple[float, ...],
        crop_max_scales: Tuple[float, ...],
        transforms: T.Compose,
    ):
        crop_transforms = []
        for i in range(len(crop_sizes)):
            random_resized_crop = T.RandomResizedCrop(
                crop_sizes[i], scale=(crop_min_scales[i], crop_max_scales[i])
            )

            crop_transforms.extend(
                [
                    T.Compose(
                        [
                            random_resized_crop,
                            transforms,
                        ]
                    )
                ]
                * crop_counts[i]
            )
        super().__init__(crop_transforms)

class SwaVTransform(MultiCropTranform):
    def __init__(
        self,
        crop_sizes: Tuple[int, int] = (11, 8),
        crop_counts: Tuple[int, int] = (2, 3),
        crop_min_scales: Tuple[float, float] = (0.8, 0.2),
        crop_max_scales: Tuple[float, float] = (1.0, 0.5),

        normalize: Union[None, Dict[str, List[float]]] = None,
    ):

        transforms = T.Compose([
            T.Normalize(mean=normalize['mean'], std=normalize['std'])
            ])

        super().__init__(
            crop_sizes=crop_sizes,
            crop_counts=crop_counts,
            crop_min_scales=crop_min_scales,
            crop_max_scales=crop_max_scales,
            transforms=transforms,
        )
  