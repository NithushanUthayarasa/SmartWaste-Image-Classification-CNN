# 🗑️ Garbage Image Classification using Deep Learning (CNN & Computer Vision)

**End-to-end AI project for automated multi-class garbage image classification using Convolutional Neural Networks (CNNs) and computer vision techniques.**  
Leverages deep learning and transfer learning to accurately classify images into 10 garbage categories, demonstrating practical applications of computer vision in real-world waste management.


---

## 📌 Project Overview
Effective waste segregation is critical for recycling and sustainability. Manual sorting is inefficient and error-prone.  

This project demonstrates how **deep learning can automate waste classification**, improving recycling efficiency and reducing operational costs.  

**Pipeline includes:**
- Image preprocessing & normalization  
- CNN model training using transfer learning  
- Evaluation on unseen test data  
- Prediction visualization and error analysis  
- Classification metrics and confusion matrix  
- Production-ready saved model and reproducible workflow  

---

## 🚨 Problem Statement
Improper waste segregation leads to:
- Reduced recycling efficiency  
- Increased environmental pollution  
- Higher operational costs  

Traditional rule-based systems fail due to **high visual variability** in waste materials.  

**AI Solution:** Frame waste segregation as a **multi-class image classification problem** using CNNs to learn visual patterns from images.

---

## 🎯 Objectives
- Automate waste classification using computer vision  
- Improve recycling efficiency with AI-driven sorting  
- Demonstrate applied AI skills in CNNs and transfer learning  
- Build a portfolio-ready, production-style ML project  

---

## 🧾 Dataset
**Source:** [Garbage Classification V2 – Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)  
**License:** MIT (educational & research purposes)  

| Attribute | Value |
|-----------|-------|
| Total Images | ~20,000+ |
| Classes | 10 |
| Categories | battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash |

**Preprocessing:** Images were resized to 224×224, normalized, and split into train, validation, and test sets.

---

## 🧠 Model Architecture
- **Input Size:** 224 × 224  
- **Architecture:** CNN (Transfer Learning-based)  
- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** RMSprop  
- **Evaluation Strategy:** Hold-out test set, no data leakage  

---

## 🔄 End-to-End Pipeline
1. Dataset loading & directory structuring  
2. Image preprocessing & normalization  
3. Train / validation / test split  
4. CNN model training with checkpointing  
5. Model evaluation on test data  
6. Prediction visualization  
7. Confusion matrix & classification report  
8. Export predictions to CSV  

---

## 🧪 Evaluation Metrics
- Accuracy  
- Precision, Recall, F1-score (per class)  
- Confusion Matrix  

---

## 📊 Results (Test Set)
**Test Accuracy:** 88.6%  
**Test Loss:** 0.35  

**Key Observations:**
- Strong performance on `clothes`, `biological`, and `shoes`  
- Moderate confusion between `metal`, `trash`, and `plastic`  
- Balanced performance across all 10 classes  

**Classification Report (Precision / Recall / F1-Score):**

| Class      | Precision | Recall | F1-score |
|------------|----------|--------|----------|
| Battery    | 0.90     | 0.83   | 0.86     |
| Biological | 0.95     | 0.95   | 0.95     |
| Cardboard  | 0.86     | 0.89   | 0.88     |
| Clothes    | 0.97     | 0.98   | 0.98     |
| Glass      | 0.90     | 0.87   | 0.88     |
| Metal      | 0.66     | 0.82   | 0.73     |
| Paper      | 0.81     | 0.81   | 0.81     |
| Plastic    | 0.84     | 0.77   | 0.81     |
| Shoes      | 0.94     | 0.97   | 0.95     |
| Trash      | 0.77     | 0.71   | 0.74     |

---

## 🔍 Confusion Matrix
Misclassifications mainly occur between **visually similar categories** (metal, plastic, trash).  
This confirms the CNN effectively captures **texture, shape, and visual features** for classification.

---

## 🚀 Deployment
- **Saved Model:** `models/best_model.keras`  
- Batch inference supported  
- Fully reproducible preprocessing & evaluation workflow  
- Predictions exported as `predictions.csv` for analysis  

---
```text
## 📁 Project Structure
cv-waste-classification/
│
├── data/
│ ├── raw/ # Original Kaggle dataset
│ ├── processed/ # Train / validation / test splits
│
├── models/
│ └── best_model.keras
│
├── notebooks/
│ ├── data_preprocessing.ipynb
│ ├── model_training.ipynb
│ ├── evaluation.ipynb
│
├── reports/
│ ├── confusion_matrix.png
│ └── metrics_summary.csv
│
├── predictions.csv
└── README.md

```

---

## 🌟 Highlights
- ✅ End-to-end **CNN & deep learning project** with transfer learning  
- 📊 Achieved 88.6% test accuracy on 10 waste categories  
- 🧠 Strong interpretability using confusion matrix & classification report  
- ⚙️ Production-ready saved model with **reproducible AI pipeline**  
- 🔁 Demonstrates **practical AI, deep learning, and computer vision skills**

---

## 🛠 Tech Stack
- Python 3.10+  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## 🌍 Business & Social Impact
- Automated waste segregation  
- Improved recycling efficiency  
- Environmental sustainability  
- Potential applications in **smart cities & IoT waste management**  

---

## 👤 Author
**Nithushan Uthayarasa**  
BSc (Hons) in Information Technology – Specialized in **Artificial Intelligence (AI)**  

