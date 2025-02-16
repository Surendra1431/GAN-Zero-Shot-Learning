# 🎨 GAN-Zero-Shot Learning (GAN-ZSL)  
🚀 *Generating Synthetic Features for Unseen Classes Using Wasserstein GANs*

This project explores **Zero-Shot Learning (ZSL)** using **Wasserstein Generative Adversarial Networks (WGAN-GP)**. By synthesizing feature representations for **unseen classes**, this method enables classifiers to predict categories without direct supervision. 🧠✨

---

## 📝 Overview

Traditional supervised learning relies on **labeled samples** for all classes, which is impractical for real-world applications where **new classes** may appear frequently.  

**Zero-Shot Learning (ZSL)** overcomes this limitation by transferring knowledge from **seen classes** to **unseen classes** using **semantic attributes**.  

This project uses **Generative Adversarial Networks (GANs)** to **synthesize feature representations** for unseen categories, allowing classifiers to recognize novel objects **without direct training data**. 🦅🔥

---

## ⚡ Key Features
1. 🏆 **WGAN-GP for Zero-Shot Learning** – Generate synthetic features for unseen classes.  
2. 🖼 **Feature Extraction using ResNet-101** – Utilize deep feature representations for training.  
3. 🎯 **One-vs-All Classifier** – Predict unseen classes based on generated features.  
4. 📊 **t-SNE Visualization** – Compare synthetic and real features graphically.  
5. 🤖 **Hyperparameter Tuning** – Optimize WGAN-GP and classification models for best performance.  

---

## 📊 Dataset

This project is built on the **CUB-200-2011** dataset, a well-known benchmark for Zero-Shot Learning.

🔹 **Dataset Features:**
- 🐦 **200 bird species (classes)**
- 📸 **11,788 images**
- 🔠 **Attribute-based class descriptions**
- ✅ **Seen Classes** – Used for model training  
- ❓ **Unseen Classes** – Used for Zero-Shot prediction  

📌 **Download Dataset**: [Caltech Birds Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)  

💾 **Preprocessed Features**: Extracted using **ResNet-101**, stored in the `/features/` directory.

---

## 🤖 Model Architecture

### 🎨 1️⃣ Feature Extraction
- Uses **ResNet-101 (pretrained on ImageNet)** to extract **2048-dimensional** feature vectors.

### 🎲 2️⃣ WGAN-GP Generator
- **Input**: `(Random noise + Attribute vector)`
- **Output**: `Synthetic feature vectors`
- **Activation**: `LeakyReLU, Tanh`

### 🏗 3️⃣ WGAN-GP Discriminator
- **Input**: `(Feature vector + Attribute vector)`
- **Output**: `Real or Fake classification`
- **Loss**: Wasserstein Loss with Gradient Penalty

### 🎯 4️⃣ One-vs-All Classifier
- **Input**: `Feature vectors`
- **Output**: `Predicted class labels`

### 🔑 5️⃣ Attribute Mapping Model
- **Input**: `Feature vectors`
- **Output**: `Predicted class attributes`
![image](https://github.com/user-attachments/assets/4a7b592d-4535-47cb-964f-8d164d8791f7)
---

## 🛠 Installation & Setup

### 📦 Prerequisites
Make sure you have **Python 3.7+** installed. Then install the required dependencies:

```bash
pip install tensorflow scikit-learn tqdm matplotlib seaborn keras-tuner


