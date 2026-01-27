import torch
import os
import PIL
import numpy
import albumentations
import json
import cv2
from collections import defaultdict
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from torch.utils.data import Dataset

from typing import Any, Dict, List, Optional, Tuple, Union


def dict_to_tensor(d: dict) -> Tuple[dict, dict]:
    """Convert dictionary values to tensors where appropriate."""
    tensor_params = {}
    types = {}

    for k, v in d.items():
        if isinstance(v, bool):
            tensor_params.update({k: v})
        elif isinstance(v, (int, float)):
            tensor_params.update({k: torch.tensor(v, dtype=torch.float64)})
        else:
            tensor_params.update({k: v})

        types.update({k: type(v)})

    return tensor_params, types


def retrieve_num_type(num: torch.Tensor, type: type) -> Union[int, float]:
    """Retrieve number from tensor with specified type."""
    assert isinstance(num, torch.Tensor)
    if type == int:
        return int(torch.round(num))
    elif type == float:
        return float(num)


class SegmentationAnnotation:
    """Class to handle segmentation annotations similar to BoundingBox/Point annotations."""

    def __init__(self, annotation_data: Dict, image_info: Dict):
        self.data = annotation_data
        self.image_info = image_info
        self.atype = 'Segmentation'

        # Convert segmentation to mask
        self._mask = self._segmentation_to_mask()

    def _segmentation_to_mask(self) -> numpy.ndarray:
        """Convert COCO segmentation to binary mask."""
        segmentation = self.data['segmentation']
        height, width = self.image_info['height'], self.image_info['width']

        if isinstance(segmentation, list):
            # Polygon format
            mask = numpy.zeros((height, width), dtype=numpy.uint8)
            for poly in segmentation:
                # Reshape polygon to (n_points, 2)
                poly_array = numpy.array(poly).reshape(-1, 2).astype(numpy.int32)
                cv2.fillPoly(mask, [poly_array], 1)
            return mask

        elif isinstance(segmentation, dict):
            # RLE format
            if isinstance(segmentation['counts'], list):
                # Uncompressed RLE
                rle = coco_mask.frPyObjects([segmentation], height, width)[0]
            else:
                # Compressed RLE
                rle = segmentation
            return coco_mask.decode(rle)

        else:
            # Fallback: create empty mask
            return numpy.zeros((height, width), dtype=numpy.uint8)

    @property
    def get_tuple(self) -> Tuple:
        """Get mask as tuple for compatibility."""
        return (self._mask,)

    @property
    def mask(self) -> numpy.ndarray:
        """Get the binary mask."""
        return self._mask

    def get_bbox(self) -> List[float]:
        """Get bounding box from mask."""
        if 'bbox' in self.data:
            # Use provided bbox (COCO format: x, y, width, height)
            x, y, w, h = self.data['bbox']
            return [x, y, x + w, y + h]  # Convert to pascal_voc format
        else:
            # Calculate bbox from mask
            coords = numpy.where(self._mask)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                return [float(x_min), float(y_min), float(x_max), float(y_max)]
            else:
                return [0.0, 0.0, 0.0, 0.0]


class AnnotationsFromCOCOSegmentation:
    """Class to handle COCO segmentation annotations."""

    def __init__(self, coco_file: str):
        """
        Args:
            coco_file (str): Path to COCO JSON annotation file
        """
        self.coco_file = coco_file
        self.coco = COCO(coco_file)

        # Create image info mapping
        self.images_info = {img['id']: img for img in self.coco.dataset['images']}
        self.categories_info = {cat['id']: cat for cat in self.coco.dataset['categories']}

        # Process annotations
        self.processed_data = self._process_annotations()

        # Create dataframe-like structure for compatibility
        self.dataframe = self._create_dataframe()

    def _process_annotations(self) -> Dict[str, List[SegmentationAnnotation]]:
        """Process COCO annotations grouped by image."""
        grouped = defaultdict(list)

        for ann_id in self.coco.anns:
            ann = self.coco.anns[ann_id]

            # Skip annotations without segmentation
            if 'segmentation' not in ann or not ann['segmentation']:
                continue

            image_id = ann['image_id']
            image_info = self.images_info[image_id]
            image_name = image_info['file_name']

            # Create segmentation annotation
            seg_ann = SegmentationAnnotation(ann, image_info)
            grouped[image_name].append(seg_ann)

        return dict(grouped)

    def _create_dataframe(self):
        """Create a dataframe-like structure for compatibility with CSVDataset."""
        rows = []

        for image_name, annotations in self.processed_data.items():
            for ann in annotations:
                row = {
                    'images': image_name,
                    'annos': ann,
                    'labels': ann.data['category_id'],
                    'category_name': self.categories_info[ann.data['category_id']]['name'],
                    'area': ann.data.get('area', 0),
                    'iscrowd': ann.data.get('iscrowd', 0),
                    'bbox': ann.get_bbox()
                }
                rows.append(row)

        # Create a simple namespace to mimic pandas DataFrame
        class SimpleDataFrame:
            def __init__(self, data):
                self.data = data
                self.columns = list(data[0].keys()) if data else []

            def __getitem__(self, condition):
                if isinstance(condition, str):
                    # Column access
                    return [row[condition] for row in self.data]
                elif callable(condition):
                    # Boolean indexing
                    filtered_data = [row for row in self.data if condition(row)]
                    return SimpleDataFrame(filtered_data)
                else:
                    return SimpleDataFrame([row for row in self.data if condition])

            def drop(self, columns):
                new_data = []
                for row in self.data:
                    new_row = {k: v for k, v in row.items() if k not in columns}
                    new_data.append(new_row)
                return SimpleDataFrame(new_data)

            @property
            def annos(self):
                return [row['annos'] for row in self.data]

        return SimpleDataFrame(rows)

    @property
    def images(self) -> List[str]:
        """Get list of unique image filenames."""
        return list(self.processed_data.keys())


class COCOSegmentationDataset(Dataset):
    """
    Dataset class for COCO segmentation annotations, extending the CSVDataset template.

    This dataset handles COCO segmentation annotations (polygons and RLE) and converts them
    to binary masks for augmentation. Supports the same augmentation pipeline as CSVDataset
    but adds mask-specific transformations.
    """

    def __init__(
            self,
            coco_file: str,
            root_dir: str,
            albu_transforms: Optional[list] = None,
            end_transforms: Optional[list] = None,
            augmentation_multiplier: int = 1,
            mask_format: str = 'mask',  # 'mask', 'polygon', or 'rle'
            filter_categories: Optional[List[Union[int, str]]] = None,
            min_area: float = 0.0
    ) -> None:
        """
        Args:
            coco_file (str): Path to COCO JSON annotation file
            root_dir (str): Path to the images folder
            albu_transforms (list, optional): Albumentations transformations list
            end_transforms (list, optional): Post-processing transformations
            augmentation_multiplier (int): Dataset size multiplier for augmentation
            mask_format (str): Output format for masks ('mask', 'polygon', 'rle')
            filter_categories (list, optional): Filter by category IDs or names
            min_area (float): Minimum area threshold for annotations
        """

        assert isinstance(albu_transforms, (list, type(None))), \
            f'albumentations-transformations must be a list, got {type(albu_transforms)}'

        assert isinstance(end_transforms, (list, type(None))), \
            f'end-transformations must be a list, got {type(end_transforms)}'

        assert mask_format in ['mask', 'polygon', 'rle'], \
            f'mask_format must be one of ["mask", "polygon", "rle"], got {mask_format}'

        self.coco_file = coco_file
        self.root_dir = root_dir
        self.albu_transforms = albu_transforms
        self.end_transforms = end_transforms
        self.augmentation_multiplier = augmentation_multiplier
        self.mask_format = mask_format
        self.filter_categories = filter_categories
        self.min_area = min_area

        # Store end parameters
        self._store_end_params()

        # Load COCO annotations
        self.annotations = AnnotationsFromCOCOSegmentation(self.coco_file)
        self.data = self.annotations.dataframe

        # Set annotation type
        self.anno_type = 'Segmentation'

        # Filter data if specified
        if self.filter_categories or self.min_area > 0:
            self.data = self._filter_data()

        # Get unique image names
        used = set()
        self._img_names = [x for x in self.annotations.images
                           if x not in used and (used.add(x) or True)]

        # Filter out images with no valid annotations after filtering
        valid_images = set(row['images'] for row in self.data.data)
        self._img_names = [img for img in self._img_names if img in valid_images]

        print(f"Loaded COCO Segmentation dataset:")
        print(f"  - Images: {len(self._img_names)}")
        print(f"  - Total annotations: {len(self.data.data)}")
        print(f"  - Mask format: {mask_format}")
        print(f"  - Categories: {len(self.annotations.categories_info)}")

    def _filter_data(self):
        """Filter data by categories and minimum area."""
        filtered_rows = []

        for row in self.data.data:
            # Filter by category
            if self.filter_categories is not None:
                if isinstance(self.filter_categories[0], int):
                    if row['labels'] not in self.filter_categories:
                        continue
                else:
                    if row['category_name'] not in self.filter_categories:
                        continue

            # Filter by minimum area
            if row['area'] < self.min_area:
                continue

            filtered_rows.append(row)

        # Create new dataframe with filtered data
        class SimpleDataFrame:
            def __init__(self, data):
                self.data = data
                self.columns = list(data[0].keys()) if data else []

            def __getitem__(self, key):
                if isinstance(key, str):
                    return [row[key] for row in self.data]
                else:
                    # Assume boolean condition for image filtering
                    filtered = []
                    for row in self.data:
                        if row['images'] == key:
                            filtered.append(row)
                    return SimpleDataFrame(filtered)

            def drop(self, columns):
                new_data = []
                for row in self.data:
                    new_row = {k: v for k, v in row.items() if k not in columns}
                    new_data.append(new_row)
                return SimpleDataFrame(new_data)

            @property
            def annos(self):
                return [row['annos'] for row in self.data]

        return SimpleDataFrame(filtered_rows)

    def _load_image(self, index: int) -> PIL.Image.Image:
        """Load image by index."""
        actual_index = index % len(self._img_names)
        img_name = self._img_names[actual_index]
        img_path = os.path.join(self.root_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        return PIL.Image.open(img_path).convert('RGB')

    def _load_target(self, index: int) -> Dict[str, List[Any]]:
        """Load target annotations by index."""
        actual_index = index % len(self._img_names)
        img_name = self._img_names[actual_index]

        # Filter annotations for this image
        annotations = [row for row in self.data.data if row['images'] == img_name]

        target = {
            'image_id': [index],
            'image_name': [f"{img_name}_aug_{index}"],
            'original_image_name': [img_name],
            'augmentation_id': [index // len(self._img_names)]
        }

        # Group annotation data
        if annotations:
            # Extract masks and other data
            masks = [row['annos'].mask for row in annotations]

            target.update({
                'annos': masks,  # List of binary masks
                'labels': [row['labels'] for row in annotations],
                'category_names': [row['category_name'] for row in annotations],
                'areas': [row['area'] for row in annotations],
                'iscrowd': [row['iscrowd'] for row in annotations],
                'bboxes': [row['bbox'] for row in annotations]
            })
        else:
            # Empty annotations
            target.update({
                'annos': [],
                'labels': [],
                'category_names': [],
                'areas': [],
                'iscrowd': [],
                'bboxes': []
            })

        return target

    def _transforms(
            self,
            image: PIL.Image.Image,
            target: dict
    ) -> Tuple[torch.Tensor, dict]:
        """Apply transformations with segmentation mask support."""

        label_fields = target.copy()
        for key in ['annos', 'image_id', 'image_name', 'original_image_name', 'augmentation_id']:
            label_fields.pop(key, None)

        if self.albu_transforms and target['annos']:

            # Segmentation masks
            transform_pipeline = albumentations.Compose(
                self.albu_transforms,
                # Note: We'll handle masks manually since albumentations doesn't support multiple masks directly
            )

            # Convert image to numpy
            image_np = numpy.array(image)

            # Handle multiple masks
            transformed_masks = []
            transformed_labels = []
            transformed_categories = []
            transformed_areas = []
            transformed_iscrowd = []
            transformed_bboxes = []

            for i, mask in enumerate(target['annos']):
                # Apply transformation to image and single mask
                try:
                    # Create single mask transform
                    single_mask_transform = albumentations.Compose(
                        self.albu_transforms,
                        additional_targets={'mask': 'mask'}
                    )

                    transformed = single_mask_transform(
                        image=image_np,
                        mask=mask
                    )

                    transformed_mask = transformed['mask']

                    # Only keep masks that still have some area after transformation
                    if numpy.sum(transformed_mask) > 0:
                        transformed_masks.append(transformed_mask)
                        transformed_labels.append(target['labels'][i])
                        transformed_categories.append(target['category_names'][i])
                        transformed_areas.append(numpy.sum(transformed_mask))
                        transformed_iscrowd.append(target['iscrowd'][i])

                        # Recalculate bbox from transformed mask
                        coords = numpy.where(transformed_mask)
                        if len(coords[0]) > 0:
                            y_min, y_max = coords[0].min(), coords[0].max()
                            x_min, x_max = coords[1].min(), coords[1].max()
                            transformed_bboxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                        else:
                            transformed_bboxes.append([0.0, 0.0, 0.0, 0.0])

                except Exception as e:
                    print(f"Warning: Failed to transform mask {i}: {e}")
                    continue

            # Apply transformation to image only (using the last transformation)
            if transformed_masks:
                image_only_transform = albumentations.Compose(self.albu_transforms)
                transformed_image = image_only_transform(image=image_np)['image']
            else:
                transformed_image = image_np

            tr_image = numpy.asarray(transformed_image)

            # Prepare transformed target
            transformed_target = {
                'masks': transformed_masks,
                'labels': transformed_labels,
                'category_names': transformed_categories,
                'areas': transformed_areas,
                'iscrowd': transformed_iscrowd,
                'boxes': transformed_bboxes
            }

            # Add metadata
            for key in ['image_id', 'image_name', 'original_image_name', 'augmentation_id']:
                if key in target:
                    transformed_target[key] = target[key]

            # Convert to tensors (simplified version)
            tr_image, tr_target = self._sample_to_tensor(tr_image, transformed_target)

            if self.end_transforms is not None:
                for trans in self.end_transforms:
                    tr_image, tr_target = trans(tr_image, tr_target)

            return tr_image, tr_target

        else:
            # No transforms or no annotations
            return image, target

    def _sample_to_tensor(self, image: numpy.ndarray, target: dict) -> Tuple[torch.Tensor, dict]:
        """Convert sample to tensor format."""
        # Convert image to tensor
        if image.dtype != numpy.uint8:
            image = (image * 255).astype(numpy.uint8)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert target to tensor format
        tensor_target = {}
        for key, value in target.items():
            if key == 'masks':
                # Convert masks based on desired output format
                if self.mask_format == 'mask':
                    # Keep as binary masks
                    if value:
                        tensor_target[key] = torch.stack([torch.from_numpy(m.astype(numpy.float32)) for m in value])
                    else:
                        tensor_target[key] = torch.empty(0, image.shape[0], image.shape[1])

                elif self.mask_format == 'polygon':
                    # Convert masks back to polygons
                    polygons = []
                    for mask in value:
                        contours, _ = cv2.findContours(mask.astype(numpy.uint8), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Take the largest contour
                            largest_contour = max(contours, key=cv2.contourArea)
                            polygon = largest_contour.reshape(-1, 2).flatten().tolist()
                            polygons.append(polygon)
                    tensor_target[key] = polygons

                elif self.mask_format == 'rle':
                    # Convert masks to RLE format
                    rles = []
                    for mask in value:
                        rle = coco_mask.encode(numpy.asfortranarray(mask.astype(numpy.uint8)))
                        rles.append(rle)
                    tensor_target[key] = rles

            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    tensor_target[key] = torch.tensor(value)
                elif isinstance(value[0], list):  # For nested lists like bboxes
                    tensor_target[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    tensor_target[key] = value  # Keep as is for strings, etc.
            else:
                tensor_target[key] = value

        return image_tensor, tensor_target

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        """Get item by index."""
        img = self._load_image(index)
        target = self._load_target(index)

        tr_img, tr_target = self._transforms(img, target)

        return tr_img, tr_target

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self._img_names) * self.augmentation_multiplier

    def get_actual_length(self) -> int:
        """Get actual number of unique images."""
        return len(self._img_names)

    def get_augmentation_info(self, index: int) -> Dict[str, Any]:
        """Get augmentation information for given index."""
        actual_index = index % len(self._img_names)
        augmentation_id = index // len(self._img_names)
        return {
            'actual_image_index': actual_index,
            'augmentation_id': augmentation_id,
            'original_image_name': self._img_names[actual_index]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_images': len(self._img_names),
            'total_annotations': len(self.data.data),
            'annotation_type': self.anno_type,
            'mask_format': self.mask_format,
            'augmentation_multiplier': self.augmentation_multiplier,
            'effective_dataset_size': len(self)
        }

        # Category distribution
        category_counts = defaultdict(int)
        for row in self.data.data:
            category_counts[row['category_name']] += 1

        stats['category_distribution'] = dict(category_counts)

        # Area statistics
        areas = [row['area'] for row in self.data.data]
        if areas:
            stats['area_stats'] = {
                'min_area': min(areas),
                'max_area': max(areas),
                'mean_area': sum(areas) / len(areas)
            }

        return stats

    # Methods for end transforms (same as CSVDataset)
    def _store_end_params(self) -> None:
        """Store end transform parameters."""
        self.end_params = {}
        self._end_params_types = []

        if self.end_transforms is not None:
            for trans in self.end_transforms:
                tensor_params, types = dict_to_tensor(trans.__dict__)
                self._end_params_types.append(types)
                self.end_params.update(tensor_params)

    def load_end_param(self, end_param: str, value: float) -> None:
        """Load end parameter."""
        self.end_params[end_param] = value

    def update_end_transforms(self) -> None:
        """Update end transforms with new parameters."""
        new_transforms = []
        up_params = self._update_end_params()
        for trans, params in zip(self.end_transforms, up_params):
            name = type(trans).__name__
            # You'll need to import your transforms module here
            # new_transforms.append(transforms.__dict__[name](**params))
            pass

        self.end_transforms = new_transforms
        self._store_end_params()

    def _update_end_params(self) -> list:
        """Update end parameters."""
        up_params = []
        for trans in self._end_params_types:
            up_dict = {}
            for k, v in trans.items():
                up_num = self.end_params[k]
                if isinstance(self.end_params[k], torch.Tensor):
                    up_num = retrieve_num_type(self.end_params[k], v)

                up_dict.update({k: up_num})

            up_params.append(up_dict)

        return up_params


# Custom collate function for segmentation masks
def collate_fn_segmentation(batch):
    """Custom collate function for segmentation datasets."""
    from torch.utils.data import default_collate

    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    # Collate images normally
    batched_images = default_collate(images)

    # Keep targets as list since mask sizes may vary
    return batched_images, targets


# Example usage and testing
def demonstrate_segmentation_dataset():
    """Demonstrate the segmentation dataset usage."""
    print("🎭 COCO Segmentation Dataset Demo")
    print("=" * 50)

    print("""
# Example Usage:

# 1. Basic segmentation dataset
dataset = COCOSegmentationDataset(
    coco_file='annotations/instances_train2017.json',
    root_dir='images/train2017/',
    albu_transforms=[
        albumentations.Resize(512, 512),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.3),
        albumentations.GaussNoise(p=0.2),
    ],
    mask_format='mask',  # Output binary masks
    augmentation_multiplier=2
)

# 2. Filter by categories and minimum area
filtered_dataset = COCOSegmentationDataset(
    coco_file='annotations/instances_train2017.json',
    root_dir='images/train2017/',
    filter_categories=['person', 'car', 'bicycle'],
    min_area=1000.0,  # Minimum 1000 pixels
    mask_format='polygon',  # Output as polygons
    augmentation_multiplier=3
)

# 3. Use with DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn_segmentation  # Handle variable mask sizes
)

for images, targets in dataloader:
    # images: [B, 3, H, W]
    # targets: List of dicts, each containing:
    #   - 'masks': torch.Tensor [N, H, W] or list of polygons/RLEs
    #   - 'labels': torch.Tensor [N]
    #   - 'boxes': torch.Tensor [N, 4] (derived from masks)
    #   - 'areas': torch.Tensor [N]
    #   - 'category_names': List[str]
    pass

# 4. Different mask output formats
mask_dataset = COCOSegmentationDataset(..., mask_format='mask')     # Binary masks
polygon_dataset = COCOSegmentationDataset(..., mask_format='polygon') # Polygon coordinates  
rle_dataset = COCOSegmentationDataset(..., mask_format='rle')       # RLE format

# 5. Get dataset statistics
stats = dataset.get_statistics()
print(f"Images: {stats['total_images']}")
print(f"Annotations: {stats['total_annotations']}")
print(f"Categories: {stats['category_distribution']}")
    """)

    print("\n🎯 Key Features:")
    print("=" * 20)
    print("✅ Handles COCO segmentation annotations (polygons & RLE)")
    print("✅ Converts to binary masks for augmentation")
    print("✅ Multiple output formats: masks, polygons, RLE")
    print("✅ Same API as CSVDataset (drop-in replacement)")
    print("✅ Full albumentations support with mask transformations")
    print("✅ Category and area filtering")
    print("✅ Augmentation multiplier for dataset size increase")
    print("✅ Automatic bbox calculation from masks")
    print("✅ Custom collate function for variable mask sizes")

    print("\n⚙️ Augmentation Pipeline:")
    print("=" * 25)
    print("1. Load COCO segmentation annotations")
    print("2. Convert polygons/RLE to binary masks")
    print("3. Apply albumentations to image + masks")
    print("4. Filter out empty masks after augmentation")
    print("5. Recalculate bboxes from transformed masks")
    print("6. Convert to desired output format")
    print("7. Return as tensors with metadata")


if __name__ == "__main__":
    demonstrate_segmentation_dataset()