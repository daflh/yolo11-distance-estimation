<div align="center">
  <h1>YOLO11-dist: Monocular Distance Estimation</h1>
  <p>Object detection and absolute distance estimation using a single monocular camera based on YOLO11</p>
</div>

---

## 🎓 About This Project

This repository contains the source code for my undergraduate thesis (*Skripsi*) at **Universitas Gadjah Mada (UGM)**.

In simple terms, this project modifies the YOLO11 object detection model so that it can estimate **how far away an object is**, not just detect it.

Normally, YOLO models answer two questions:

- **What is the object?** (classification)
- **Where is it in the image?** (bounding box)

The modified model, **YOLO11-dist**, adds a third one:

- **How far away is the object in the real world?**

And it does this using **only a single RGB camera**.

### Why this matters

Many safety systems such as **ADAS (Advanced Driver Assistance Systems)** need accurate distance information to avoid collisions.  
However, most reliable distance sensing solutions rely on sensors like **LiDAR**, which can cost hundreds or thousands of dollars.

If distance can be estimated reliably from a **cheap monocular camera**, safety features could become much more accessible and easier to deploy.

This project explores that possibility by integrating **absolute distance estimation directly into the YOLO detection pipeline**.

---

# 🧠 Architecture Overview

The main idea of this work is simple:

Instead of building a separate distance estimation network, **distance prediction is integrated directly into the YOLO detection head**.

YOLO normally predicts:

```

[x, y, w, h, class]

```

YOLO11-dist extends this to:

```

[x, y, w, h, class, distance]

```

Each detected object now also has an **absolute distance estimate (in meters)**.

### Modified Detection Head

YOLO11 uses a decoupled head structure for classification and localization.

This project introduces a **third prediction branch**:

```

cv2 → classification
cv3 → bounding box regression
cv4 → distance estimation   ← added

````

The new **cv4 branch** predicts the absolute distance for each detected object.

### Feature Fusion for Distance Estimation

Estimating distance from a single 2D image is inherently ambiguous.

To make the prediction more stable, the distance head does not rely solely on convolutional features.  
It also incorporates **geometric cues derived from the predicted bounding box**, specifically:

- bounding box **height**
- bounding box **diagonal length**

These geometric signals correlate strongly with distance in perspective images.

An ablation study in the thesis shows that combining:

- **raw feature maps**
- **bounding box geometry**

produces significantly better distance predictions than using features alone.

### Distance Loss Function

Distance prediction is trained using a **Weighted Mean Squared Error (WMSE)** loss.

The weighting prioritizes **closer objects**, since errors at short range are much more critical for collision avoidance.

---

![Architecture Image](replace_with_your_architecture_image_url_later.png)

---

# 📊 Dataset and Evaluation

The model is trained and evaluated using the **KITTI dataset**, one of the standard benchmarks for autonomous driving research.

Distance estimation focuses on objects within a **0–100 meter range**, which is the most relevant region for driving safety.

### Detection Performance

| Metric | Value |
|------|------|
| Precision | 91.9% |
| Recall | 86.5% |
| mAP50-95 | 0.714 |

### Distance Estimation Performance

| Metric | Value |
|------|------|
| Mean Absolute Error (MAE) | **0.981 m** |
| Mean Relative Error (MRE) | **4.48%** |

### Edge Device Performance

The model was also tested on a **Raspberry Pi 5** using the NCNN inference backend.

| Device | FPS |
|------|------|
| Raspberry Pi 5 (CPU, NCNN) | **10.7 FPS** |

This demonstrates that the approach remains **lightweight enough for embedded systems**.

---

# 📈 Comparison with Previous Methods

We compare **YOLO11n-dist** against previous monocular distance estimation approaches.

| Method | Params (M) | FLOPs (B) | MAE (m) | MRE |
|------|------|------|------|------|
| Dist-YOLO | 42.6 | N/A | 2.49 | 0.110 |
| DECADE | 3.3 | 8.7 | 1.38 | 0.073 |
| **YOLO11n-dist (Ours)** | **2.67** | **6.7** | **0.981** | **0.045** |

YOLO11n-dist achieves:

- **60.8% lower MAE than Dist-YOLO**
- **28.9% lower MAE than DECADE**

while also using **fewer parameters** than both models.

---

# 🚀 Quick Start

### Installation

Install dependencies the same way as the standard Ultralytics repository:

```bash
pip install ultralytics
````

---

## Dataset Format

The label format extends the standard YOLO format by appending the 3D distance values:

```
<class_id> <x_center> <y_center> <width> <height> <dx> <dy> <dz>
```

Where:

* `dx`, `dy`, `dz` are object distances in meters
* the model primarily learns from **dz (longitudinal distance)**

---

# Training Procedure

Training follows a **two-stage curriculum learning strategy**.

### Stage 1 — Object Detection

First train the model as a normal detector.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model.load("yolo11n.pt")

model.train(
    data="KITTI.yaml",
    epochs=200,
    imgsz=640
)
```

---

### Stage 2 — Distance Estimation

Then train the distance prediction head.

The backbone is partially frozen to preserve detection performance.

```python
model = YOLO("yolo11n-dist.yaml")
model.load("runs/train/weights/best.pt")

model.train(
    data="KITTI.yaml",
    epochs=200,
    freeze=9,
    max_dist=100.0
)
```

Distance-sensitive augmentations such as scaling and perspective transformations are disabled during this stage.

---

# Validation

Evaluate both detection and distance performance:

```python
model = YOLO("yolo11n-dist_best.pt", task="dist")

metrics = model.val(data="KITTI.yaml")
```

Metrics include:

* detection mAP
* distance MAE
* distance MRE

---

# Inference

Example inference code:

```python
from ultralytics import YOLO

model = YOLO("yolo11n-dist_best.pt", task="dist")

results = model.predict(
    source="test_video.mp4",
    device="cpu"
)

for result in results:
    boxes = result.boxes
    distances = result.distances
```

Each detection returns:

```
[class, bounding box, estimated distance]
```

---

# 📚 References

* **Ultralytics YOLO11** – base detection framework
* **KITTI Vision Benchmark Suite** – Geiger et al. (2012)

Distance estimation comparison methods:

* Dist-YOLO (2022)
* DECADE (2024)

---

# 📄 License

This project is licensed under **AGPL-3.0**, inheriting the license from the original Ultralytics YOLO repository.

============================================================

# YOLO11 Distance Estimation

Distance-aware object detection based on **YOLO11** for **monocular absolute distance estimation**.

This repository contains a modified YOLO11 architecture that performs **object detection and distance estimation simultaneously** in a single forward pass.

The model was developed as part of an undergraduate thesis focused on enabling **efficient monocular distance estimation for autonomous driving and embedded systems**.

---

# Overview

Estimating the absolute distance of objects from a **single RGB camera** is a challenging problem because depth information is not explicitly available in images.

Previous approaches such as **Dist-YOLO** and **DECADE** address this problem by either:

* extending YOLO prediction vectors, or
* using multi-stage pipelines combining detection and regression modules.

This project proposes a **modified YOLO11 architecture** where distance estimation is integrated directly into the **detection head**, allowing the model to predict:

* bounding box location
* object class
* absolute distance

simultaneously.

The approach leverages both:

* **deep feature maps from the neck**
* **geometric cues from bounding box dimensions (height and diagonal)**

which were found to have a strong correlation with object distance.

---

# Architecture

The model extends the YOLO11 detection head with an additional **distance regression branch**.

The prediction head outputs:

```
[x, y, w, h, class, distance]
```

Distance is predicted **per bounding box** for **true positive detections**.

Distance regression is trained using **Weighted Mean Squared Error (WMSE)**.

Key features:

* Integrated **multi-task learning**
* Distance prediction from:

  * feature maps
  * bounding box height
  * bounding box diagonal
* Lightweight architecture
* Only **+3.04% parameter increase** compared to the base YOLO11 model 

---

# Dataset

Training and evaluation were performed using the **KITTI dataset**.

Characteristics:

* Monocular RGB images
* Urban driving scenarios
* Distance range: **0 – 100 meters**

The training process used a **two-stage curriculum learning approach**:

Stage 1
Train standard object detection.

Stage 2
Train the distance estimation head while preserving detection performance.

---

# Results

## Detection Performance

| Metric    | Value |
| --------- | ----- |
| Precision | 91.9% |
| Recall    | 86.5% |
| mAP50     | 0.922 |
| mAP50-95  | 0.714 |

---

## Distance Estimation Performance

| Metric | Value       |
| ------ | ----------- |
| MAE    | **0.981 m** |
| MRE    | **4.48%**   |

---

## Comparison with Previous Work

| Method              | Params (M) | FLOPs (B) | MAE (m)   | MRE       |
| ------------------- | ---------- | --------- | --------- | --------- |
| Dist-YOLO           | 42.6       | N/A       | 2.50      | 0.11      |
| DECADE              | 3.3        | 8.7       | 1.38      | 0.073     |
| YOLO11n-dist (ours) | **2.67**   | **6.7**   | **0.981** | **0.045** |

The proposed model achieves:

* **28.9% lower MAE than DECADE**
* **60.8% lower MAE than Dist-YOLO**

while using **fewer parameters and FLOPs**.

---

# Embedded Deployment

The model was tested on **Raspberry Pi 5**.

Using the **NCNN inference framework**, the system achieved:

```
10.7 FPS (CPU inference)
```

This demonstrates that the model can run on **single board computers for embedded vision applications**.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/daflh/yolo11-distance-estimation
cd yolo11-distance-estimation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Example training command:

```bash
yolo train \
model=yolo11n.yaml \
data=kitti.yaml \
epochs=200
```

Training is performed in **two stages**:

1️⃣ Detection training
2️⃣ Distance estimation training

---

# Inference

Example inference:

```bash
yolo predict \
model=best.pt \
source=image.jpg
```

Example output:

```
car 0.91
distance: 14.3 m
```

---

# Project Structure

```
ultralytics/
models/
distance_error_analysis.ipynb
model_benchmark.py
test.py
```

---

# Citation

If you use this work, please cite:

```
@thesis{haq2026,
  title={Development of an Object Detection and Absolute Distance Estimation System Using Monocular Camera on Single Board Computer},
  author={Muhammad Daaffi Ul Haq},
  year={2026}
}
```

---

# Acknowledgements

* Ultralytics YOLO
* KITTI dataset
* Previous work on Dist-YOLO and DECADE

============================================================

<div align="center">
  <h1>YOLO11-dist: Monocular Distance Estimation</h1>
  <p>Object detection and absolute distance estimation using monocular camera based on YOLO11</p>
</div>

## 🎓 About This Project

[cite_start]This repository contains the source code for my undergraduate thesis (*Skripsi*) at Universitas Gadjah Mada (UGM)[cite: 1876]. 

**What does it do?** In a concrete, non-scientific way: This project takes the popular YOLO11 object detection model and makes it smarter. [cite_start]Normally, YOLO can tell you *what* an object is and *where* it is on the screen by drawing a bounding box around it[cite: 2108]. [cite_start]My modified version, **YOLO11-dist**, also tells you exactly *how far away* that object is in the real world (in meters), using just a single standard camera[cite: 2097]. 

**Why was it created?**
For self-driving cars and Advanced Driver Assistance Systems (ADAS) to avoid collisions, they need to know the distance to other vehicles and pedestrians. [cite_start]Usually, this requires expensive sensors like LiDAR, which can cost thousands of dollars[cite: 1934]. By using a regular, cheap monocular camera and AI to estimate distance, we can make life-saving safety features much more affordable and accessible.

## 🧠 The New Architecture

[cite_start]To make YOLO11 understand distance, I modified its "prediction head"—the part of the network that makes final decisions[cite: 2136]. 

* [cite_start]**Decoupled Head Addition:** YOLO11 originally has branches for class predictions (`cv2`) and bounding box distributions (`cv3`)[cite: 2140]. [cite_start]I added a new branch called `cv4` specifically for absolute distance estimation[cite: 2140].
* **Feature Fusion:** Estimating distance from a 2D image is hard. [cite_start]To help the model, the `cv4` branch doesn't just look at the raw image features from the network's neck[cite: 2141]. It also uses the geometric features of the detected bounding box (specifically the height and diagonal length, which strongly correlate with distance). [cite_start]This fusion makes the prediction much more stable[cite: 2144].
* **Custom Loss Function:** I introduced a Weighted Mean Squared Error (WMSE) loss function that gives more priority to objects that are closer to the camera, because close objects are more critical for collision avoidance.

![Architecture Image](replace_with_your_architecture_image_url_later.png)

## 📊 Benchmarks & Comparison

The model was trained and evaluated on the **KITTI dataset**, focusing on objects up to 100 meters away. [cite_start]The tests were also run on a resource-constrained **Raspberry Pi 5** to prove its efficiency[cite: 2203].

**Performance Metrics:**
* **Object Detection:** Precision: 91.9% | [cite_start]Recall: 86.5% | mAP50-95: 0.714 [cite: 2398]
* **Distance Estimation:** Mean Absolute Error (MAE): 0.981 meters | [cite_start]Mean Relative Error (MRE): 4.48% [cite: 2398]
* [cite_start]**Speed (Raspberry Pi 5 CPU using NCNN):** 10.7 FPS [cite: 2450]

**Comparison with Previous SOTA:**
[cite_start]We compared YOLO11n-dist (nano version) against previous one-stage (Dist-YOLO) and multi-stage (DECADE) approaches[cite: 2471].

| Method | Params (M) | FLOPs (B) | MAE (m) | MRE |
| :--- | :--- | :--- | :--- | :--- |
| [cite_start]**Dist-YOLO** [cite: 2471] | [cite_start]42.6 [cite: 2471] | [cite_start]N/A [cite: 2471] | [cite_start]2.49 [cite: 2471] | [cite_start]0.110 [cite: 2471] |
| [cite_start]**DECADE** [cite: 2471] | [cite_start]3.3 [cite: 2471] | [cite_start]8.7 [cite: 2471] | [cite_start]1.38 [cite: 2471] | [cite_start]0.073 [cite: 2471] |
| [cite_start]**YOLO11n-dist (Ours)** [cite: 2471] | [cite_start]2.67 [cite: 2471] | [cite_start]6.7 [cite: 2471] | [cite_start]**0.981** [cite: 2471] | [cite_start]**0.045** [cite: 2471] |

*YOLO11n-dist achieved a 60.8% reduction in MAE compared to Dist-YOLO and a 28.9% reduction compared to DECADE!*

## 🚀 Quick Start

### 1. Installation
Install the required packages just like the standard Ultralytics repo:
```bash
pip install ultralytics

```

### 2. Dataset Preparation

Format your labels similarly to standard YOLO, but append the absolute 3D distances `dx`, `dy`, `dz` (in meters) to the end of each line. We primarily use `dz` (longitudinal distance) for the distance target.

```text
<class_id> <x_center> <y_center> <width> <height> <dx> <dy> <dz>

```

### 3. Training

Training is done using a two-stage Curriculum Learning approach.

**Stage 1: Object Detection**
Train the model to detect objects first, using standard pre-trained weights.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model.load(weights="yolo11n.pt")
model.train(data="KITTI.yaml", epochs=200, imgsz=640)

```

**Stage 2: Distance Estimation**
Freeze the backbone and train the distance prediction head. Distance-sensitive augmentations (like scaling or perspective) are automatically disabled.

```python
model = YOLO("yolo11n-dist.yaml")
model.load(weights="runs/train/weights/best.pt") # Weights from Stage 1
model.train(data="KITTI.yaml", epochs=200, freeze=9, max_dist=100.0)

```

### 4. Validation

Evaluate the model's performance on your validation set:

```python
model = YOLO("yolo11n-dist_best.pt", task="dist")
metrics = model.val(data="KITTI.yaml")
# Metrics will include object detection mAP as well as distance MAE and MRE

```

### 5. Inference

Run inference on images or videos. The model will output bounding boxes along with the estimated distance in meters.

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-dist_best.pt", task="dist")
results = model.predict(source="test_video.mp4", device="cpu")

for result in results:
    boxes = result.boxes
    # Access predicted distances along with the bounding boxes
    distances = result.distances 

```

## 📜 References & Acknowledgements

* **Ultralytics YOLO11:** The foundational framework for this project.
* **KITTI Dataset:** Geiger, A., et al. "Are we ready for autonomous driving? The KITTI vision benchmark suite." (2012) .


* **Dist-YOLO:** Vajgl, M., et al. "Dist-YOLO: Fast Object Detection with Distance Estimation." (2022) .


* **DECADE:** Shahzad, M., et al. "DECADE: Towards Designing Efficient-yet-Accurate Distance Estimation Modules..." (2024).



## 📄 License

This project is licensed under the **AGPL-3.0 License**, inheriting from the original Ultralytics YOLO repository.
