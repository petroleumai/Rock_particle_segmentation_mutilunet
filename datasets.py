import os
import json
import numpy as np
import torch
from PIL import Image
from skimage.draw import polygon
from torch.utils.data import Dataset
import albumentations as A

def generate_label_from_json_binarized(json_path, original_width=2000, original_height=1450):
    """Generate binary segmentation mask from JSON annotation"""
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    label = np.zeros((original_height, original_width), dtype=np.float32)
    shapes = json_data.get('shapes', [])
    
    for shape in shapes:
        points = shape.get('points', [])
        if len(points) < 2:
            continue
        new_points = [(int(pt[0]), int(pt[1])) for pt in points]
        rr, cc = polygon([pt[1] for pt in new_points], [pt[0] for pt in new_points], shape=label.shape)
        label[rr, cc] = 1
    return label

class MultiViewRockDataset(Dataset):
    """Multi-view rock dataset with JSON annotations"""
    def __init__(self, root_folder, transform=None, num_views=5, augmentations=None):
        self.root_folder = root_folder
        self.subfolders = []
        self.transform = transform
        self.num_views = num_views
        self.augmentations = augmentations

        for sub_d in os.listdir(root_folder):
            sub_d_path = os.path.join(root_folder, sub_d)
            if os.path.isdir(sub_d_path):
                json_files = [f for f in os.listdir(sub_d_path) if f.endswith('.json')]
                if len(json_files) == 1:
                    self.subfolders.append(sub_d_path)

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, idx):
        subfolder = self.subfolders[idx]
        image_files = sorted([f for f in os.listdir(subfolder) if f.endswith(('.jpg', '.png'))])
        json_files = [f for f in os.listdir(subfolder) if f.endswith('.json')]
        
        if len(json_files) != 1:
            raise ValueError(f"Expected exactly one JSON file in {subfolder}, but found {len(json_files)}.")
        json_path = os.path.join(subfolder, json_files[0])
        original_label = generate_label_from_json_binarized(json_path)

        transformed_views = []
        for i in range(self.num_views):
            if i < len(image_files):
                img_path = os.path.join(subfolder, image_files[i])
                image = np.array(Image.open(img_path))
            else:
                image = np.zeros((256, 256, 3), dtype=np.uint8)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image, mask=original_label)
                transformed_image = transformed['image']
                transformed_label = transformed['mask']
            else:
                transformed_image = image
                transformed_label = original_label
            
            transformed_views.append(transformed_image)

        # Apply augmentations
        if self.augmentations:
            augmented_images = []
            for i in range(self.num_views):
                augmented = self.augmentations(image=transformed_views[i], mask=transformed_label)
                augmented_images.append(augmented['image'])
                transformed_label = augmented['mask']
            transformed_views = augmented_images

        # Convert to tensors
        images = np.stack(transformed_views, axis=0)
        images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        label = torch.tensor(transformed_label, dtype=torch.float32).unsqueeze(0)

        return images, label

def load_data_with_augmentation(root_folder, seed=42):
    """Load dataset with augmentation support"""
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ], additional_targets={'mask': 'mask'})
    
    augmentations = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3)
    ])
    
    dataset_original = MultiViewRockDataset(root_folder, transform=transform)
    dataset_augmented = MultiViewRockDataset(root_folder, transform=transform, augmentations=augmentations)
    
    full_dataset = torch.utils.data.ConcatDataset([dataset_original, dataset_augmented])
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)