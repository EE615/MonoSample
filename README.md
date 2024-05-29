# MonoSample: Synthetic 3D Data Augmentation Method in Monocular 3D Object Detection


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

