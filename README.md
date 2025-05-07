
# 🧬 Gene Expression Classification using KNN and SVM

This project applies machine learning techniques to classify cancer presence based on gene expression data. By analyzing the expression levels of two genes, we aim to determine whether a given patient sample indicates cancer. This analysis is particularly useful for early detection, contributing to more effective treatment strategies.

## 🔍 Overview

The dataset contains three columns:
- `Gene One` (continuous): Expression level of the first gene
- `Gene Two` (continuous): Expression level of the second gene
- `Cancer Present` (binary): 1 if cancer is present, 0 otherwise

We explore and visualize the data, preprocess it, and then apply machine learning classification techniques:
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

We also tune hyperparameters using **GridSearchCV** and assess model performance using accuracy metrics and cross-validation.

## 📊 Project Workflow

1. **Data Cleaning and Exploration**
   - Removed duplicates and checked for missing values
   - Visualized gene expression distributions
   - Plotted scatter plots and heatmaps for correlation analysis

2. **Preprocessing**
   - Categorical encoding of target variable (`0 → No`, `1 → Yes`)
   - Train-test split of the dataset

3. **Model Training & Evaluation**
   - Trained SVM and KNN models on the training set
   - Evaluated using accuracy, confusion matrix, and classification reports
   - Performed cross-validation to measure generalization performance

4. **Hyperparameter Tuning**
   - Used GridSearchCV for finding optimal `C` values and kernels in SVM

5. **Performance Benchmarking**
   - Measured training time for SVM and KNN models
   - Compared model accuracy on both training and testing sets

## 🛠️ Tech Stack

- **Python**
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Jupyter Notebook`

## 📈 Results

- Both KNN and SVM performed well in classifying cancer presence.
- Cross-validation helped validate the robustness of each model.
- Model performance metrics and timing were benchmarked for practical comparison.

## 🏷️ Badges

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Modeling-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

