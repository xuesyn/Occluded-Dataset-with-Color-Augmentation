# Occluded-Dataset-with-Color-Augmentation

This repo include codes for our work in [How does Color Constancy Affect Target Recognition and
Instance Segmentation?](https://doi.org/10.1145/3474085.3475690), an augmentation strategy which combine color augmentation and synthetic occlusion augmentation. Our method emulates wrong WB images under high occlusion level, which generally have bad performance on current DNNs for detection and segmentation.

## How to Start

### Preparation

```shell
git clone https://github.com/xuesyn/Occluded-Dataset-with-Color-Augmentation.git
cd Occluded-Dataset-with-Color-Augmentation

pip install -r requirements.txt
```

### Augmentation

```shell
python CreateOccludedDatasetWithColorAugmentation.py --input_image_dir test --output_image_dir occlusionColoredAugmentedDataset
```

## How to Augment Your Own Dataset

- prepare your dataset as follow (e.g. "test" in this repo):

> .
> ├── Annotations
> │   └── class1
> │       └── 000000000283.txt
> ├── Images
> │   └── class1
> │       └── 000000000283.jpg
> ├── Image_sets
> │   └── class1_val.txt
> └── obj_mask
>  └── class1

- change class name in "CreateOccludedDatasetWithColorAugmentation.py"

- run following order:

  ```shell
  python CreateOccludedDatasetWithColorAugmentation.py --input_image_dir trainingDataset --output_image_dir augmentedDataset
  ```

  
