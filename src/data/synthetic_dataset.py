"""Synthetic dataset for zero-shot learning experiments."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SyntheticDataset(Dataset):
    """Synthetic dataset for zero-shot learning experiments.
    
    This dataset generates synthetic images with different patterns/colors
    to simulate different classes for zero-shot learning evaluation.
    """
    
    def __init__(
        self,
        class_descriptions: List[str],
        num_samples_per_class: int = 100,
        image_size: Tuple[int, int] = (224, 224),
        num_channels: int = 3,
        split: str = "train",
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        augmentation: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        """Initialize synthetic dataset.
        
        Args:
            class_descriptions: List of class descriptions for zero-shot learning.
            num_samples_per_class: Number of samples per class.
            image_size: Size of generated images.
            num_channels: Number of image channels.
            split: Dataset split ('train', 'val', 'test').
            train_split: Fraction of data for training.
            val_split: Fraction of data for validation.
            test_split: Fraction of data for testing.
            augmentation: Data augmentation configuration.
            seed: Random seed for reproducibility.
        """
        self.class_descriptions = class_descriptions
        self.num_classes = len(class_descriptions)
        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        self.num_channels = num_channels
        self.split = split
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        
        # Calculate split indices
        total_samples = self.num_classes * self.num_samples_per_class
        train_end = int(total_samples * train_split)
        val_end = train_end + int(total_samples * val_split)
        
        if split == "train":
            self.start_idx = 0
            self.end_idx = train_end
        elif split == "val":
            self.start_idx = train_end
            self.end_idx = val_end
        else:  # test
            self.start_idx = val_end
            self.end_idx = total_samples
        
        self.length = self.end_idx - self.start_idx
        
        # Set up transforms
        self.transform = self._get_transforms(augmentation)
        
        # Generate synthetic data
        self._generate_data()
    
    def _generate_data(self) -> None:
        """Generate synthetic image data."""
        self.images = []
        self.labels = []
        
        for class_idx in range(self.num_classes):
            for sample_idx in range(self.num_samples_per_class):
                global_idx = class_idx * self.num_samples_per_class + sample_idx
                
                # Only include samples in current split
                if self.start_idx <= global_idx < self.end_idx:
                    # Generate synthetic image based on class
                    image = self._generate_class_image(class_idx, sample_idx)
                    self.images.append(image)
                    self.labels.append(class_idx)
    
    def _generate_class_image(self, class_idx: int, sample_idx: int) -> np.ndarray:
        """Generate a synthetic image for a specific class.
        
        Args:
            class_idx: Index of the class.
            sample_idx: Index of the sample within the class.
            
        Returns:
            Generated image as numpy array.
        """
        height, width = self.image_size
        
        # Create base pattern based on class
        if class_idx == 0:  # Cat-like pattern
            image = self._create_cat_pattern(height, width, sample_idx)
        elif class_idx == 1:  # Dog-like pattern
            image = self._create_dog_pattern(height, width, sample_idx)
        elif class_idx == 2:  # Car-like pattern
            image = self._create_car_pattern(height, width, sample_idx)
        elif class_idx == 3:  # Tree-like pattern
            image = self._create_tree_pattern(height, width, sample_idx)
        elif class_idx == 4:  # Bird-like pattern
            image = self._create_bird_pattern(height, width, sample_idx)
        elif class_idx == 5:  # Horse-like pattern
            image = self._create_horse_pattern(height, width, sample_idx)
        elif class_idx == 6:  # Ship-like pattern
            image = self._create_ship_pattern(height, width, sample_idx)
        elif class_idx == 7:  # Truck-like pattern
            image = self._create_truck_pattern(height, width, sample_idx)
        elif class_idx == 8:  # Bicycle-like pattern
            image = self._create_bicycle_pattern(height, width, sample_idx)
        else:  # Person-like pattern
            image = self._create_person_pattern(height, width, sample_idx)
        
        # Add noise for variation
        noise = np.random.normal(0, 0.1, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _create_cat_pattern(self, height: int, width: int, sample_idx: int) -> np.ndarray:
        """Create cat-like pattern."""
        image = np.zeros((height, width, 3))
        
        # Cat body (oval shape)
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Body
        body_mask = ((x_coords - center_x) / (width * 0.3)) ** 2 + \
                   ((y_coords - center_y) / (height * 0.4)) ** 2 <= 1
        image[body_mask] = [0.8, 0.6, 0.4]  # Orange color
        
        # Ears
        ear_y = center_y - height * 0.3
        ear_x1, ear_x2 = center_x - width * 0.15, center_x + width * 0.15
        ear_mask1 = ((x_coords - ear_x1) / (width * 0.08)) ** 2 + \
                   ((y_coords - ear_y) / (height * 0.1)) ** 2 <= 1
        ear_mask2 = ((x_coords - ear_x2) / (width * 0.08)) ** 2 + \
                   ((y_coords - ear_y) / (height * 0.1)) ** 2 <= 1
        image[ear_mask1 | ear_mask2] = [0.8, 0.6, 0.4]
        
        return image
    
    def _create_dog_pattern(self, height: int, width: int, sample_idx: int) -> np.ndarray:
        """Create dog-like pattern."""
        image = np.zeros((height, width, 3))
        
        # Dog body (rectangular shape)
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Body
        body_mask = (abs(x_coords - center_x) <= width * 0.25) & \
                   (abs(y_coords - center_y) <= height * 0.3)
        image[body_mask] = [0.6, 0.4, 0.2]  # Brown color
        
        # Head
        head_mask = ((x_coords - center_x) / (width * 0.2)) ** 2 + \
                   ((y_coords - center_y + height * 0.2) / (height * 0.25)) ** 2 <= 1
        image[head_mask] = [0.6, 0.4, 0.2]
        
        return image
    
    def _create_car_pattern(self, height: int, width: int, sample_idx: int) -> np.ndarray:
        """Create car-like pattern."""
        image = np.zeros((height, width, 3))
        
        # Car body (rectangular)
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Body
        body_mask = (abs(x_coords - center_x) <= width * 0.3) & \
                   (abs(y_coords - center_y) <= height * 0.15)
        image[body_mask] = [0.2, 0.2, 0.8]  # Blue color
        
        # Wheels
        wheel_y = center_y + height * 0.1
        wheel_x1, wheel_x2 = center_x - width * 0.2, center_x + width * 0.2
        wheel_mask1 = ((x_coords - wheel_x1) / (width * 0.05)) ** 2 + \
                     ((y_coords - wheel_y) / (height * 0.05)) ** 2 <= 1
        wheel_mask2 = ((x_coords - wheel_x2) / (width * 0.05)) ** 2 + \
                     ((y_coords - wheel_y) / (height * 0.05)) ** 2 <= 1
        image[wheel_mask1 | wheel_mask2] = [0.1, 0.1, 0.1]  # Black wheels
        
        return image
    
    def _create_tree_pattern(self, height: int, width: int, sample_idx: int) -> np.ndarray:
        """Create tree-like pattern."""
        image = np.zeros((height, width, 3))
        
        # Tree trunk
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Trunk
        trunk_mask = (abs(x_coords - center_x) <= width * 0.05) & \
                    (y_coords >= center_y)
        image[trunk_mask] = [0.4, 0.2, 0.1]  # Brown trunk
        
        # Leaves (circular)
        leaves_mask = ((x_coords - center_x) / (width * 0.25)) ** 2 + \
                     ((y_coords - center_y + height * 0.1) / (height * 0.25)) ** 2 <= 1
        image[leaves_mask] = [0.2, 0.6, 0.2]  # Green leaves
        
        return image
    
    def _create_bird_pattern(self, height: int, width: int, sample_idx: int) -> np.ndarray:
        """Create bird-like pattern."""
        image = np.zeros((height, width, 3))
        
        # Bird body (small oval)
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Body
        body_mask = ((x_coords - center_x) / (width * 0.1)) ** 2 + \
                   ((y_coords - center_y) / (height * 0.15)) ** 2 <= 1
        image[body_mask] = [0.8, 0.8, 0.2]  # Yellow color
        
        # Wings
        wing_mask = ((x_coords - center_x) / (width * 0.2)) ** 2 + \
                   ((y_coords - center_y) / (height * 0.1)) ** 2 <= 1
        image[wing_mask] = [0.8, 0.8, 0.2]
        
        return image
    
    def _create_horse_pattern(self, height: int, width: int, sample_idx: int) -> np.ndarray:
        """Create horse-like pattern."""
        image = np.zeros((height, width, 3))
        
        # Horse body
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Body
        body_mask = (abs(x_coords - center_x) <= width * 0.2) & \
                   (abs(y_coords - center_y) <= height * 0.25)
        image[body_mask] = [0.6, 0.3, 0.1]  # Brown color
        
        # Head
        head_mask = ((x_coords - center_x) / (width * 0.15)) ** 2 + \
                   ((y_coords - center_y + height * 0.25) / (height * 0.2)) ** 2 <= 1
        image[head_mask] = [0.6, 0.3, 0.1]
        
        return image
    
    def _create_ship_pattern(self, height: int, width: int, sample_idx: int) -> np.ndarray:
        """Create ship-like pattern."""
        image = np.zeros((height, width, 3))
        
        # Ship hull
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Hull
        hull_mask = (abs(x_coords - center_x) <= width * 0.3) & \
                   (y_coords >= center_y - height * 0.1)
        image[hull_mask] = [0.3, 0.3, 0.3]  # Gray color
        
        # Mast
        mast_mask = (abs(x_coords - center_x) <= width * 0.02) & \
                   (y_coords <= center_y - height * 0.1)
        image[mast_mask] = [0.4, 0.2, 0.1]  # Brown mast
        
        return image
    
    def _create_truck_pattern(self, height: int, width: int, sample_idx: int) -> np.ndarray:
        """Create truck-like pattern."""
        image = np.zeros((height, width, 3))
        
        # Truck body
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Body
        body_mask = (abs(x_coords - center_x) <= width * 0.25) & \
                   (abs(y_coords - center_y) <= height * 0.2)
        image[body_mask] = [0.8, 0.2, 0.2]  # Red color
        
        # Wheels
        wheel_y = center_y + height * 0.15
        wheel_x1, wheel_x2 = center_x - width * 0.15, center_x + width * 0.15
        wheel_mask1 = ((x_coords - wheel_x1) / (width * 0.06)) ** 2 + \
                     ((y_coords - wheel_y) / (height * 0.06)) ** 2 <= 1
        wheel_mask2 = ((x_coords - wheel_x2) / (width * 0.06)) ** 2 + \
                     ((y_coords - wheel_y) / (height * 0.06)) ** 2 <= 1
        image[wheel_mask1 | wheel_mask2] = [0.1, 0.1, 0.1]  # Black wheels
        
        return image
    
    def _create_bicycle_pattern(self, height: int, width: int, sample_idx: int) -> np.ndarray:
        """Create bicycle-like pattern."""
        image = np.zeros((height, width, 3))
        
        # Bicycle frame
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Frame lines
        frame_mask = (abs(x_coords - center_x) <= width * 0.02) | \
                    (abs(y_coords - center_y) <= height * 0.02)
        image[frame_mask] = [0.2, 0.2, 0.2]  # Black frame
        
        # Wheels
        wheel_y = center_y
        wheel_x1, wheel_x2 = center_x - width * 0.2, center_x + width * 0.2
        wheel_mask1 = ((x_coords - wheel_x1) / (width * 0.08)) ** 2 + \
                     ((y_coords - wheel_y) / (height * 0.08)) ** 2 <= 1
        wheel_mask2 = ((x_coords - wheel_x2) / (width * 0.08)) ** 2 + \
                     ((y_coords - wheel_y) / (height * 0.08)) ** 2 <= 1
        image[wheel_mask1 | wheel_mask2] = [0.1, 0.1, 0.1]  # Black wheels
        
        return image
    
    def _create_person_pattern(self, height: int, width: int, sample_idx: int) -> np.ndarray:
        """Create person-like pattern."""
        image = np.zeros((height, width, 3))
        
        # Person silhouette
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Head
        head_mask = ((x_coords - center_x) / (width * 0.08)) ** 2 + \
                   ((y_coords - center_y + height * 0.3) / (height * 0.08)) ** 2 <= 1
        image[head_mask] = [0.8, 0.6, 0.4]  # Skin color
        
        # Body
        body_mask = (abs(x_coords - center_x) <= width * 0.06) & \
                   (abs(y_coords - center_y) <= height * 0.2)
        image[body_mask] = [0.2, 0.2, 0.8]  # Blue shirt
        
        # Arms
        arm_mask = (abs(x_coords - center_x) <= width * 0.15) & \
                  (abs(y_coords - center_y + height * 0.1) <= height * 0.03)
        image[arm_mask] = [0.8, 0.6, 0.4]  # Skin color
        
        return image
    
    def _get_transforms(self, augmentation: Optional[Dict[str, Any]] = None) -> transforms.Compose:
        """Get image transforms.
        
        Args:
            augmentation: Augmentation configuration.
            
        Returns:
            Composed transforms.
        """
        transform_list = []
        
        # Convert to PIL Image
        transform_list.append(transforms.ToPILImage())
        
        # Add augmentation if specified
        if augmentation and augmentation.get("enabled", False):
            if augmentation.get("horizontal_flip", 0) > 0:
                transform_list.append(
                    transforms.RandomHorizontalFlip(p=augmentation["horizontal_flip"])
                )
            
            if augmentation.get("rotation", 0) > 0:
                transform_list.append(
                    transforms.RandomRotation(degrees=augmentation["rotation"])
                )
            
            if any(augmentation.get(key, 0) > 0 for key in ["brightness", "contrast", "saturation"]):
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=augmentation.get("brightness", 0),
                        contrast=augmentation.get("contrast", 0),
                        saturation=augmentation.get("saturation", 0),
                    )
                )
        
        # Resize and normalize
        transform_list.extend([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get dataset item.
        
        Args:
            idx: Item index.
            
        Returns:
            Tuple of (image, label).
        """
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_descriptions(self) -> List[str]:
        """Get class descriptions for zero-shot learning.
        
        Returns:
            List of class descriptions.
        """
        return self.class_descriptions
