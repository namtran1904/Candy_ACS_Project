# 🍬 ACS (Automatic Candy Selector)

An end-to-end Computer Vision and Deep Learning project designed to automatically detect, classify, and sort candies using a robust cascade architecture. Developed as part of the **IN450** advanced algorithm course at **Esisar (Grenoble INP - UGA)**.

---

## 📂 Project Structure & Data Pipeline

1. **Data Pipeline (`extract_frames.py` & `split_dataset.py`):**
   - Automatically extracts video frames and resizes them to a standard format ($256\times256$ pixels).
   - Shuffles and splits the dataset into **80% Training** and **20% Testing** sets.

2. **Model Training & Comparison (Classification Task):**
   - We tested and benchmarked multiple Deep Learning architectures to identify candy types and colors on white backgrounds:
     - **VGG-16:** Heavy and low accuracy (~74%).
     - **MobileNetV3:** Lightweight and fast (~93%).
     - **ResNet-18 & ResNet-50:** Excellent performance, stable, and fast (~93% to 99%).
     - **Vision Transformer (ViT):** Highly accurate (~98%), but memory-intensive.
   - **Regularization & Augmentation:** Applied Data Augmentation (rotation, flipping, lighting adjustments) alongside `weight_decay` and `AdamW` optimizers to prevent overfitting.

3. **Scientific Evaluation:**
   - Evaluated models using **Confusion Matrices** and classification metrics (Precision, Recall, F1-score) via `scikit-learn` and `seaborn` to analyze prediction errors (e.g., color confusion between similar shades).

4. **Global Architecture (Cascade Approach for Complex Backgrounds):**
   - To handle complex detection environments (Task 2), we implemented a two-stage cascade system:
     - **Stage 1 (Detector):** **RT-DETR** locates and crops each candy bounding box on complex backgrounds.
     - **Stage 2 (Classifier):** **ResNet-18** precisely identifies the type and color of each cropped candy.

---

## 🛠️ Tech Stack
- **Deep Learning Framework:** PyTorch, Torchvision
- **Object Detection:** RT-DETR (Ultralytics / Custom pipeline)
- **Data Analysis & Viz:** Scikit-learn, Seaborn, OpenCV, Matplotlib
- **Deployment Plan:** Exporting trained PyTorch models to **ONNX format** for seamless Java application integration.
