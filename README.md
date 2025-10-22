# 🔒 Detection of Poisoned Data in Deep Learning Models

##  Project Overview
This project is part of the **"Sécurité des systèmes d’intelligence artificielle"** module.  
It focuses on detecting **poisoned data** within deep learning models trained on the **CIFAR-10** dataset using a **probabilistic anomaly detection approach**.

In this study, we investigated the vulnerability of deep learning models to data poisoning attacks and evaluated a probabilistic detection pipeline for identifying poisoned samples.  
We demonstrated that even a well-performing CNN can be significantly impacted when a relatively small fraction of training data is poisoned.

Specifically, introducing 750 poisoned samples in both the airplane and ship classes (15% of each class) led to noticeable drops in per-class accuracy and overall model performance, highlighting the real-world risks of such attacks.

To counter this, we developed a feature-based anomaly detection method, leveraging the penultimate-layer activations of the CNN. By fitting a class-wise multivariate Gaussian on a small subset of clean samples, we modeled the natural distribution of each class’s features.  
Computing log-likelihoods for all samples enabled us to flag low-likelihood points as potentially poisoned, with thresholds determined by the 5th percentile of clean likelihoods.

---

## 🧩 Methodology

### 1. CNN Architecture
We implemented a **Convolutional Neural Network (CNN)** for CIFAR-10 classification with the following components:

#### 🧱 Convolutional Layers
| Layer | Input → Output Filters | Kernel | Activation | Pooling |
|--------|-------------------------|----------|-------------|----------|
| Conv1 | 3 → 64 | 3×3 | ReLU + BatchNorm | MaxPool(2×2) |
| Conv2 | 64 → 128 | 3×3 | ReLU + BatchNorm | MaxPool(2×2) |
| Conv3 | 128 → 256 | 3×3 | ReLU + BatchNorm | MaxPool(2×2) |
| Conv4 | 256 → 128 | 3×3 | ReLU + BatchNorm | MaxPool(2×2) |
| Conv5 | 128 → 128 | 3×3 | ReLU + BatchNorm | MaxPool(2×2) |

#### 🔗 Fully Connected Layers
| Layer | Input → Output |
|--------|----------------|
| FC1 | 128 → 512 |
| FC2 | 512 → 256 |
| FC3 | 256 → 128 |
| FC4 | 128 → 64 |
| FC5 | 64 → 10 (output) |

#### ⚙️ Training Setup
- Optimizer: **Adam**
- Learning Rate: **0.001**
- Batch Size: **128**
- Epochs: **50**
- Loss Function: **Cross-Entropy Loss**
- Device: **GPU (Google Colab)**

---

### 2. Probabilistic Detection Pipeline
1. **Feature Extraction**: Extract features from the **penultimate layer** (before output) of the CNN.  
2. **Clean Reference Modeling**: Select 500 trusted clean samples per class and fit a **Multivariate Gaussian** (mean μₖ, covariance Σₖ).  
3. **Regularization**: Add a small term ϵ to Σₖ for numerical stability.  
4. **Log-Likelihood Computation**
5. **Thresholding**: Samples below the **5th percentile** of clean likelihoods are flagged as **suspected poisoned**.  
6. **Evaluation Metrics**: Precision, Recall, and F1-score.

---

## 📊 Results

### ✈️ Airplane Class
| Metric | Value |
|--------|--------|
| Total samples | 5,000 |
| True poisoned | 750 |
| Suspected poisoned | 783 |
| True Positives | 552 |
| False Positives | 231 |
| False Negatives | 198 |
| True Negatives | 4,019 |
| Precision | **70.5%** |
| Recall | **73.6%** |
| F1-score | **72.0%** |

### 🚢 Ship Class
| Metric | Value |
|--------|--------|
| Total samples | 5,000 |
| True poisoned | 750 |
| Suspected poisoned | 669 |
| True Positives | 408 |
| False Positives | 261 |
| False Negatives | 342 |
| True Negatives | 3,989 |
| Precision | **61.0%** |
| Recall | **54.4%** |
| F1-score | **57.5%** |

✅ The probabilistic pipeline successfully detected a large proportion of poisoned samples, confirming its interpretability and efficiency.

---

## 📁 Project Structure
```
📦 CIFAR10_Poison_Detection
├── CIFAR_10_Clean.ipynb              # Training clean CNN model
├── CIFAR_10_Poisoned.ipynb           # Training with poisoned dataset
├── Data_poisoning_Detection.ipynb    # Probabilistic detection pipeline
├── clean_model/                      # Contains clean model and weights (.pth)
│   ├── model.pth
│   └── weights.pth
├── poisoned_model/                   # Contains poisoned model and weights (.pth)
│   ├── model.pth
│   └── weights.pth
├── Report.pdf                        # Full academic report
└── README.md                         # Project description
```

---

## 🧪 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/CIFAR10_Poison_Detection.git
cd CIFAR10_Poison_Detection

# 2. Install dependencies

# 3. Run the notebooks sequentially
#    (CIFAR_10_Clean → CIFAR_10_Poisoned → Data_poisoning_Detection)
jupyter notebook
```

---

## 💡 Key Insights
- A small fraction (15%) of poisoned data can strongly degrade CNN performance.  
- Class-wise Gaussian modeling of clean features can efficiently detect poisoned samples.  
- The **5th percentile threshold** balances precision and recall effectively.  
- This probabilistic approach is interpretable, scalable, and model-agnostic.

---

##  Future Work
- Extend to **backdoor attacks** (trigger-based poisoning).  
- Combine with **autoencoder-based** reconstruction errors.  
- Apply on larger datasets and **transformer architectures**.  
- Integrate detection into **training pipelines** for real-time monitoring.

---

