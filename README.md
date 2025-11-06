# 🩺 E-HSE: Entropy-Guided Hybrid Stacking Ensemble for PCOS Detection

## 📋 Overview

This project implements an **Entropy-Guided Hybrid Stacking Ensemble (E-HSE)** framework for accurate and explainable **Polycystic Ovary Syndrome (PCOS)** detection. The model combines advanced machine learning techniques with dual explainability mechanisms to provide clinically interpretable predictions.

---

## 🎯 Key Features

### 1. **Intelligent Data Preprocessing**
- Categorical features encoded and continuous features normalized via **MinMaxScaler**
- Non-predictive identifiers removed to ensure unbiased learning

### 2. **Dual Feature Selection**
- **Boruta algorithm** used for relevance-based feature elimination
- **SHAP values** calculated to determine feature importance
- Intersection of both ensures retention of only clinically relevant features

### 3. **Entropy-Guided Stacking Ensemble**
- **Base Learners:** Random Forest, SVM, XGBoost
- **Meta Learner:** Neural Network (Multi-Layer Perceptron)
- **Shannon Entropy** quantifies each base learner's uncertainty
- Weight assigned inversely proportional to entropy — improving ensemble stability

### 4. **Dual Explainability**
- **Global Interpretation:** SHAP explains model-wide feature influence
- **Local Interpretation:** LIME explains feature impact for individual predictions

---

## 🧠 Model Architecture

```
Input Data
   │
   ├── Data Preprocessing & Normalization
   │
   ├── Dual Feature Selection (Boruta + SHAP)
   │
   ├── Base Learners: RF, SVM, XGBoost
   │        │
   │        └── Entropy-based Weighting
   │
   ├── Neural Network Meta-Learner
   │
   └── Dual Explainability: SHAP (Global) + LIME (Local)
```

---

## 🧰 Technologies and Libraries

| Category | Libraries / Tools |
|----------|------------------|
| **Data Handling** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Feature Selection** | boruta, shap |
| **Machine Learning** | scikit-learn, xgboost |
| **Explainability** | shap, lime |
| **Model Fusion** | neural network (MLP) |
| **Evaluation** | accuracy_score, f1_score, roc_auc_score |

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

```bash
# Step 1: Clone the repository
git clone https://github.com/MUKILAN0608/EHSE-PCOS-Detection.git
cd EHSE-PCOS-Detection

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the main notebook
jupyter notebook "E_HSE_PCOS_Classification.ipynb"
```

---

## 📁 Repository Structure

```
📁 E-HSE-PCOS-Detection
│
├── 📄 pcos_classification(3).ipynb   # Main implementation notebook
├── 📄 requirements.txt                   # Python dependencies
├── 📄 README.md                          # Project documentation
└── 📁 dataset/                           # Input dataset (optional link or reference)
```

---

## 📊 Evaluation Protocol

- **Validation:** 5-Fold Stratified Cross-Validation
- **Metrics Used:** Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Explainability:** SHAP for feature-level contribution, LIME for instance-level insights
- **Calibration:** Evaluated using reliability and confidence curves

---

## 🧩 Key Equations

### 1. Entropy (Uncertainty) Measurement

$$H(p) = -\sum_{c} p_c \log(p_c)$$

### 2. Entropy-based Weighting

$$w_i = \frac{1 - H_i}{\sum_{j=1}^{m} (1 - H_j)}$$

### 3. Weighted Ensemble Output

$$z = \sum_i w_i \cdot h_i(x)$$

### 4. Meta-Learner Prediction

$$y = g(z; \theta)$$

---

## 🚀 Highlights

- ✅ Combines ensemble learning and explainable AI (XAI)
- ✅ Achieves strong generalization and model transparency
- ✅ Provides clinically interpretable feature contributions
- ✅ Designed for scalable and ethical AI in healthcare

---

## 💡 Future Work

- Integration with real-time clinical decision-support systems
- Expansion to multi-center, longitudinal medical datasets
- Incorporation of lifestyle and genetic data
- Deployment through a web-based diagnostic interface

---

## 📬 Contact

For questions or collaboration opportunities, please reach out via GitHub issues or contact the repository maintainer.

**GitHub:** [MUKILAN0608](https://github.com/MUKILAN0608)

---

## ⚖️ License

This project is licensed under the **MIT License**. Free to use for academic, research, and non-commercial purposes.

---

## 🙏 Acknowledgments

Special thanks to the medical community and data science researchers whose work has inspired this project. This framework aims to bridge the gap between advanced AI and practical clinical applications.

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
