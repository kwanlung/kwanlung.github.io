+++
date = '2025-04-15T00:18:56+08:00'
draft = false
title = 'From CNN to ViT: Improving Facial Emotion Recognition'
tags = ["AI", "Deep Learning", "Computer Vision", "Project Showcase"]
description = 'Detailed journey of building a highly accurate 7-class facial emotion recognition model using Vision Transformers (ViT), custom dataset engineering, and advanced training optimization techniques.'
author= ["deanngkwanlung"]
+++

# 🧠 Motivation
My initial attempt at facial expression recognition, completed during my university studies, involved a convolutional neural network (CNN). Despite limited resources—a modest GTX 1050 GPU and sparse datasets—I managed to achieve an accuracy between 65-75%. Although promising, it wasn't sufficient for real-world applications, particularly for something as sensitive as mental health diagnostics.

Fast forward to today: armed with greater experience, advanced hardware (RTX 3060), and modern ML frameworks, I revisited this challenge using Vision Transformers (ViT). The goal was clear: surpass 80% accuracy and deepen my expertise in state-of-the-art deep learning techniques.

---

# 📊 Objective
Build and deploy a robust 7-class facial expression recognition system capable of accurately identifying fundamental human emotions:

- Anger

- Disgust

- Fear

- Happiness

- Neutral

- Sadness

- Surprise

---

# 🖼️ Image Processing & Embedding Systems

To enable high-performing ViT-based emotion classification, the system depends heavily on rigorous image processing and efficient visual embedding strategies.

### 📦 Preprocessing Pipeline
Each image undergoes a well-designed set of operations:

- **Face Region Isolation:** All datasets are either pre-aligned (RAF-DB, AffectNet) or assume center-cropped faces.

#### 📌 Transformations Applied (PyTorch)
Training augmentations were applied via `torchvision.transforms`, using the following logic:

- **Resize:** All images resized to 224x224 (ViT input)
- **Random Horizontal Flip:** Helps the model generalize across left/right facial symmetry
- **Random Rotation (±10°):** Adds rotation invariance, useful for slightly tilted faces
- **Color Jitter:** Adjusts brightness, contrast, and saturation for lighting diversity
- **Normalization:** Converts RGB to standardized values in range `[-1, 1]`

<details>
<summary>PyTorch Implementation</summary>

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(...),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
```
</details>

#### ✂️ CutMix Augmentation
Beyond traditional transforms, I integrated CutMix, a powerful augmentation technique that:

- Cuts a patch from one image and pastes it into another

- Mixes the labels proportionally

- Helps the model learn spatial invariance and improves generalization under occlusion or noise

This was particularly effective for small classes like fear and disgust.

> CutMix is especially useful when training ViTs, as they handle global patterns better than CNNs and benefit more from mixed-structure inputs.


### 🧠 Embedding System: Vision Transformer

- The input image is embedded into a patch-level sequence using `patch_size=16`.

- Each patch is linearly projected and positional embeddings are added before being fed into the transformer encoder.

- Outputs are globally pooled and passed into a linear classification head.

ViT thus converts `(B, 3, 224, 224)` into a sequence `(B, N_patches, D)` which is finally pooled into `(B, D)` before classification.

---

# 🏗️ Dataset Engineering
Effective data handling was crucial to the project's success. Typical facial expression recognition (FER) datasets suffer from noise, inconsistencies, and imbalance. I tackled these issues directly:



| Dataset     | Link                                                                 | Notes                          |
|-------------|----------------------------------------------------------------------|--------------------------------|
| FER+        | [ferplus-7cls](https://huggingface.co/datasets/deanngkl/ferplus-7cls)         | Reduced and standardized to seven fundamental emotions   |
| AffectNet   | [affectnet_no_contempt](https://huggingface.co/datasets/deanngkl/affectnet_no_contempt) | Removed the less relevant 'contempt' class to align with the project's emotion set       |
| RAF-DB      | [raf-db-7emotions](https://huggingface.co/datasets/deanngkl/raf-db-7emotions)      | Addressed multiple data issues by manually augmenting neutral expressions from FER+ and standardizing labels    |

These enhanced datasets collectively provided 75,398 training and 8,377 validation samples, offering sufficient variety and balance to effectively train the model.

---

# 🧪 Advanced Model Training
The backbone for this project was ViT-Tiny (patch16_224) provided by timm, offering powerful performance even under hardware constraints. Key elements of my optimized training workflow included:

- Optimizer: AdamW for adaptive gradient handling.

- Scheduler: Cosine annealing learning rate with warmup phases, aiding convergence and stability.

- Augmentation: Implemented CutMix and horizontal flipping, significantly improving generalization.

- Mixed Precision Training (AMP): Enabled faster computations and reduced memory footprint, crucial for efficient GPU utilization.

## Sample Training Workflow:
```bash
python train.py
```

## Interactive Model Deployment:
```bash
uvicorn app:app --host localhost --port 8000 --reload
```

---

# Ⓜ️ Model
[ViT-Tiny FER @ Hugging Face](https://huggingface.co/deanngkl/vit-tiny-fer)

---

# 📊 Metrics
[Tensorboard Logs](https://huggingface.co/deanngkl/vit-tiny-fer/tensorboard)

---

# 📈 Performance & Insights
The model achieved a remarkable 82.2% validation accuracy, with balanced precision and recall across emotions, notably excelling in recognizing happiness (93.4%) and neutrality (91.8%). This demonstrates ViT's effectiveness in handling complex visual patterns compared to traditional CNN-based methods.

- Key insights:

    - ViT-Tiny is excellent for constrained environments, providing near state-of-the-art accuracy.

    - Meticulous dataset preparation significantly enhances model performance.

    - Data augmentation strategies like CutMix, paired with cosine LR scheduling, lead to robust learning.

    - Publicly sharing datasets enhances reproducibility and fosters community engagement.

---

# 🔮 Future Directions
To further push the boundaries of this project, the following advancements are planned:

- Scaling Up: Experiment with ViT-Base architecture combined with LoRA fine-tuning to capture more intricate visual patterns.

- PHQ Score Integration: Incorporate a regression head to predict Patient Health Questionnaire (PHQ) scores from facial expressions, directly targeting depression detection.

- Interactive Demos: Create user-friendly interfaces using Streamlit or Gradio for broader accessibility.

---

# 🚧 Challenges & Solutions
- Dataset Complexity: Integrating multiple disparate datasets was non-trivial. Custom preprocessing scripts resolved inconsistencies and improved data quality.

- Resource Constraints: Opting for ViT-Tiny balanced model complexity and hardware limitations (RTX 3060).

- Regression Task (PHQ): Initially planned, the regression task required extensive multimodal video data. Limited access led to postponement, but groundwork has been laid for future integration.

---

# 🙌 Final Thoughts
This project symbolizes my journey from academic exploration to professional-grade ML engineering. It highlights the iterative nature of machine learning—each step refines understanding and enhances technical capability. For those transitioning into AI, embracing incremental improvement is essential.  

Explore the complete project source on GitHub ([face-vit-phq](https://github.com/kwanlung/face-vit-phq)) and follow my continuing journey in AI on [Deanhub](https://kwanlung.github.io/).

---

# 📬 Connect with Me
<b>Dean Ng Kwan Lung</b>  
Blog        : [Portfolio](https://kwanlung.github.io/)  
LinkedIn    : [LinkedIn](https://www.linkedin.com/in/deanng00/)  
GitHub      : [GitHub](https://github.com/kwanlung)  
Email       : kwanlung123@gmail.com