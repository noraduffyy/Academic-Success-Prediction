# Academic Success Prediction: Model Comparison and Feature Selection

## Overview

This project explores predictive modeling of student academic outcomes—**Graduate**, **Dropout**, or **Enrolled**—using a dataset with both demographic and academic features. The goal was to assess how various machine learning models perform on this classification task and how feature selection strategies (including Information Gain and demographic-only features) affect model performance.

---

## Dataset

The dataset, `academic_success.csv`, consists of **4424 student records** with **35 columns**, including:

- Demographic attributes
- Academic performance metrics
- Economic indicators
- A target variable (`Target`)

---

## Methodology

### 1. Data Preprocessing

- Read the dataset using `pandas`
- Inspected column names and target class distribution
- Performed a **stratified train-test split** to maintain class proportions across training and test sets

### 2. Feature Selection via Information Gain

- Implemented custom functions for entropy and information gain
- Ranked features by their information gain with respect to the target
- Selected the **top 10 features** with the highest information gain for reduced-feature experiments

### 3. Model Training

Used three classification models:
- **Logistic Regression**
- **Random Forest Classifier**
- **HistGradientBoostingClassifier**

Each model was trained and evaluated using:
- All features
- Top 10 information gain features
- Only **demographic features** (a manually selected subset)

### 4. Evaluation Metrics

Model performance was evaluated using:
- **Accuracy**
- **F1 Score (weighted)**
- **Precision (weighted)**

### 5. Visualization

Created:
- Bar charts comparing model performance by input type (regular, info gain, demographic)
- Individual breakdowns by model category

---

## Key Findings

| Model                                 | Accuracy | F1 (weighted) | Precision (weighted) |
|--------------------------------------|----------|---------------|-----------------------|
| Logistic Regression (All Features)   | 0.756    | 0.740         | 0.737                |
| Logistic Regression (Info Gain)      | 0.740    | 0.722         | 0.718                |
| Random Forest (All Features)         | 0.772    | 0.754         | 0.753                |
| Random Forest (Info Gain)            | 0.745    | 0.736         | 0.732                |
| HistGradientBoosting (All Features)  | 0.768    | 0.764         | 0.763                |
| HistGradientBoosting (Info Gain)     | 0.751    | 0.749         | 0.749                |
| Logistic Regression (Demographic)    | 0.756    | 0.740         | 0.737                |
| Random Forest (Demographic)          | 0.485    | 0.466         | 0.455                |
| HistGradientBoosting (Demographic)   | 0.515    | 0.487         | 0.477                |

- **HistGradientBoosting (all features)** yielded the best overall F1 score.
- **Demographic-only models** underperformed, indicating academic performance metrics are stronger predictors.
- **Top-10 Info Gain features** preserved relatively strong performance while reducing dimensionality.

---

## Folder Structure

