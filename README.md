# MonoSample: Synthetic 3D Data Augmentation Method in Monocular 3D Object Detection

## Abstract
In the context of autonomous driving, it is both critical and challenging to locate 3D objects by using a calibrated RGB image.
Current methods typically utilize heteroscedastic aleatoric uncertainty loss to regress the depth of objects, thereby reducing the impact of noisy input while also ensuring the reliability of depth predictions.
However, experimentation reveals that uncertainty loss can also lead to serious overfitting issue and performance degradation.
To address this issue, we propose MonoSample, an augmentation method that collects samples from the dataset and places them randomly during training.
MonoSample takes into account the occlusion relationships and applies strict restrictions to ensure the verisimilitude of the enhanced scenes.
Furthermore, MonoSample avoids the complex conversion process between 2D and 3D, thereby enabling the extraction of a large number of samples and efficient operation.
Experiments on different models have verified its effectiveness.
Leveraging MonoSample in DID-M3D, our model achieves state-of-the-art (SOTA) performance on the KITTI 3D object detection benchmark.

## Introduction
The code implementation and sample library download link containing MonoSample are organized as follows:

1. **`aug` Folder**:
    - This folder contains some operators from `openpcdet`.

2. **`tools/sample_util.py`**:
    - This is where the code implementation of MonoSample is located.

3. **`notebooks` Folder**:
    - This folder contains the process for establishing the sample database and the preprocessing process of the sampling range.

4. **`decode_helper.py`**:
    - Besides, we made the following modifications to DID-M3D: we chose depth uncertainty as the score. 
    - `score = min(dets[i, j, -1], 1.)`

## Download

- [baidu](https://pan.baidu.com/s/1lTsX2ao6kuM0IWwZ8PM2cg), password: tu5y
- [google](https://drive.google.com/drive/folders/1A9yi8UUuDd0OZRYx4OrZw3TyrODE3sZP?usp=drive_link)

```angular2html

DID-M3D
├── data
│   │── KITTI3D
|   │   │── training
|   │   │   ├──calib & label_2 & image_2 & depth_dense & grid_planes & patchwork & planes
|   │   │── testing
|   │   │   ├──calib & image_2
│   │── (Specified by `database_dir` in `config.yaml`)
|   │   │── kitti_car_database_with_flip.pkl
|   │   │── sample_depth_dense_database_with_flip.pkl
|   │   │── sample_image_database_with_flip.pkl
```

## Proformance in KITTI test set

| **Benchmark**           | **Easy** | **Moderate** | **Hard** |
|-------------------------|----------|--------------|----------|
| Car (Detection)         | 96.45 %  | 95.02 %      | 85.58 %  |
| Car (Orientation)       | 96.30 %  | 94.69 %      | 85.10 %  |
| Car (3D Detection)      | 28.63 %  | 18.05 %      | 15.19 %  |
| Car (Bird's Eye View)   | 37.64 %  | 23.94 %      | 20.46 %  |

## Proformance in KITTI val set

| **Methods**                                       | **3D@IoU=0.7 Easy** | **3D@IoU=0.7 Mod.** | **3D@IoU=0.7 Hard** | **BEV@IoU=0.7 Easy** | **BEV@IoU=0.7 Mod.** | **BEV@IoU=0.7 Hard** |
|--------------------------------------------------|---------------------|---------------------|---------------------|----------------------|----------------------|----------------------|
| DID-M3D                        | 24.84               | 16.14               | 13.67               | 31.60                | 22.22                | 18.68                |
| + MonoSample                                     | **27.60**           | **19.06**           | **16.15**           | **35.10**            | **24.69**            | **21.78**            |


## Acknowledgements

This respository is mainly based on [DID-M3D](https://github.com/SPengLiang/DID-M3D) and it also benefits from [pathwork](https://github.com/LimHyungTae/patchwork), [rc-pda](https://github.com/longyunf/rc-pda?tab=readme-ov-file), [KITTI-Instance-segmentation](https://github.com/HeylenJonas/KITTI3D-Instance-Segmentation-Devkit), [openpcdet](https://github.com/open-mmlab/OpenPCDet). Thanks for their great works!