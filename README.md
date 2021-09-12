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

###### Downloading testing occluders from [OccludedPASCAL3D+](https://github.com/Angtian/OccludedPASCAL3D)

```shell
chmod +x download_occluder_lib.sh
./download_occluder_lib.sh

```

### Augmentation

```shell
python CreateOccludedDatasetWithColorAugmentation.py --input_image_dir test --output_image_dir occlusionColoredAugmentedDataset
```

## How to Augment Your Own Dataset

- prepare your dataset as follow (e.g. "test" in this repo):

> .
> 
> ├── Annotations
> 
> │   └── class1
> 
> │       └── 000000000283.txt
> 
> ├── Images
> 
> │   └── class1
> 
> │       └── 000000000283.jpg
> 
> ├── Image_sets
> 
> │   └── class1_val.txt
> 
> └── obj_mask
> 
>  └── class1

- change class name in "CreateOccludedDatasetWithColorAugmentation.py"

- run following order:

  ```shell
  python CreateOccludedDatasetWithColorAugmentation.py --input_image_dir trainingDataset --output_image_dir augmentedDataset
  ```
  
## Reference
[1] Siyan Xue, Shaobing Gao, Minjie Tan, Zhen He, and Liangtian He. 2021. How does Color Constancy Affect Target Recognition and Instance Segmentation?. In Proceedings of the 29th ACM International Conference on Multimedia (MM ’21), October 20–24, 2021, Virtual Event, China. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3474085.3475690

[2] Mahmoud Afifi and Michael S Brown. 2019. What else can fool deep learning? Addressing color constancy errors on deep neural network performance. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 243–252.

[3] Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S Brown. 2019. When color constancy goes wrong: Correcting improperly white-balanced images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1535–1544.

[4] Shao-Bing Gao, Ming Zhang, and Yong-Jie Li. 2019. Improving color constancy by selecting suitable set of training images. Optics express 27, 18 (2019), 25611–25633.
