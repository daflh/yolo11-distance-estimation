<div align="center">
  <h1>YOLO11-dist: Efficient Monocular Distance Estimation</h1>
</div>

Normally, YOLO models answer two questions:

- What is the object? (classification)
- Where is it in the image? (bounding box)

This modified model, YOLO11-dist, adds a third one:

- How far away is the object in the real world?

And it does this using only a single RGB camera.

<img src="./assets/Result_on_Validation_Dataset_with_BEV.png" alt="Result on Validation Dataset with BEV" style="width:700px; height:auto;">

# 🔎 Overview

Many safety systems such as ADAS (Advanced Driver Assistance Systems) need accurate distance information to avoid collisions. However, most reliable distance sensing solutions rely on sensors like LiDAR, which can cost hundreds or thousands of dollars. If distance can be estimated reliably from a relatively cheaper monocular camera, safety features could become much more accessible and easier to deploy.

However, estimating the absolute distance of objects from a single RGB camera is a challenging problem because depth information is not explicitly available in images. Previous approaches such as Dist-YOLO and DECADE address this problem by either:

* extending YOLO's prediction architecture, or
* using multi-models approach.

This project proposes a modified YOLO11 architecture where distance estimation is integrated directly into the detection head, allowing the model to predict:

* bounding box location
* object class
* absolute distance

simultaneously.

The approach leverages both feature maps directly from the neck as well as geometric cues from predicted bounding box dimensions (height and diagonal), which were found to have a strong correlation with object distance.

# 🧠 Architecture

The main idea of this work is simple:

Instead of building a separate distance estimation network, distance prediction is integrated directly into the YOLO detection head.

YOLO11 uses a decoupled head structure for classification and localization. This project introduces a third prediction branch:

* cv2 → object classification
* cv3 → bounding box regression
* cv4 → **distance estimation (new)**

The new cv4 branch predicts the absolute distance (in meters) for each detected object.

<img src="./assets/YOLO11-dist_Head_Architecture.png" alt="YOLO11-dist Detection Head Architecture Diagram" style="width:400px; height:auto;">

### Feature Fusion for Distance Estimation

Estimating distance from a single 2D image is inherently ambiguous.

To make the prediction more stable, the distance head does not rely solely on convolutional features.  
It also incorporates geometric cues derived from the predicted bounding box, specifically:

- bounding box height
- bounding box diagonal length

These geometric signals correlate strongly with distance in perspective images. We showed that combining raw feature maps and bounding box geometry produces significantly better distance predictions than using features alone.

### Distance Loss Function

Distance prediction is trained using a **Weighted Mean Squared Error (WMSE)** loss.

The weighting prioritizes closer objects, since errors at short range are much more critical for collision avoidance.

<img src="./assets/YOLO11-dist_Loss_Function.png" alt="YOLO11-dist Loss Function Diagram" style="width:450px; height:auto;">

# 📊 Dataset and Evaluation

The model is trained and evaluated using the **KITTI** dataset.

Distance estimation focuses on objects within a 0–100 meter range, which is the most relevant region for driving safety.

<img src="./assets/Metrics_Result_on_Training_Stage_2.png" alt="Metrics Result on Training Stage 2" style="width:500px; height:auto;">

### Detection Performance

| Metric | Value |
|------|------|
| Precision | 91.9% |
| Recall | 86.5% |
| mAP50 | 0.922 |
| mAP50-95 | 0.714 |

### Distance Estimation Performance

| Metric | Value |
|------|------|
| Mean Absolute Error (MAE) | 0.981 m |
| Mean Relative Error (MRE) | 4.48% |

### Edge Device Performance

The model was also tested on a Raspberry Pi 5 using the NCNN inference backend.

| Device | FPS |
|------|------|
| Raspberry Pi 5 (CPU) | 10.7 FPS |

This demonstrates that the approach remains lightweight enough for embedded systems.

# 📈 Comparison with Previous Methods

We compare the result against previous monocular distance estimation approaches.

| Method | Params (M) | FLOPs (B) | MAE (m) | MRE |
|------|------|------|------|------|
| Dist-YOLO | 42.6 | N/A | 2.49 | 0.110 |
| DECADE | 3.3 | 8.7 | 1.38 | 0.073 |
| YOLO11n-dist (Ours) | 2.67 | 6.7 | 0.981 | 0.045 |

YOLO11n-dist achieves:

- 60.8% lower MAE than Dist-YOLO
- 28.9% lower MAE than DECADE

while also using fewer parameters than both models.

# 🚀 Quick Start

## Installation

Clone the repository, then install dependencies:

```bash
git clone https://github.com/daflh/yolo11-distance-estimation.git
cd yolo11-distance-estimation
pip install .
```

## Preparing Dataset

The label format extends the standard YOLO format by appending the 3D distance values:

```
<class_id> <x_center> <y_center> <width> <height> <dx> <dy> <dz>
```

Where `dx`, `dy`, and `dz` are object distances for each axis in meters.

Download the labels for KITTI [here](https://drive.google.com/file/d/1-D9LIaXshfEx0d8GTLnDTCd40lbG6o0O/view?usp=sharing). You can download the dataset itself in [KITTI's official website](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

## Usage

Since the code is based on [Ultralytics YOLO original repository](https://github.com/ultralytics/ultralytics/tree/v8.3.220), please refer to the official documentation for usage.

We have provided some ready-to-use examples for training, validation, and inference in the `test.py` file. For starters, you can download the pretrained model's weight [here](https://drive.google.com/file/d/1260VcWKK-u07abYkcPQzJqdFcBwC60AX/view?usp=sharing).

# 🙏 Acknowledgements

This research was developed as part of my undergraduate thesis (*skripsi*) at Universitas Gadjah Mada (UGM) under the supervision of Dr. M. Idham Ananta Timur, M.Kom.

I would also like to acknowledge the Ultralytics team for developing and maintaining the YOLO framework that made this work possible, as well as the researchers behind the KITTI dataset.
