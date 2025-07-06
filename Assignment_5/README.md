 House Price Prediction - Week 5 Assignment (CSI Program by Celebal)

This repository contains the solution for **Week 5** of the **CSI Program** conducted by **Celebal Technologies**. The task involves predicting house prices using machine learning techniques with advanced data preprocessing and feature engineering.

##Dataset

The dataset used is from the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) Kaggle competition.

- `train.csv`: Contains training data including `SalePrice`.
- `test.csv`: Contains test data where predictions need to be made.

> Download the dataset from Kaggle and place both `train.csv` and `test.csv` in the working directory.

---

##Problem Statement

To predict the **Sale Price** of residential homes in Ames, Iowa, using various house features by applying:

- Advanced data preprocessing
- Feature engineering
- Model building using **XGBoost**

---

##  Workflow

1. **Data Loading**
   - Import and inspect `train.csv` and `test.csv`.

2. **Preprocessing**
   - Handling missing values using domain knowledge and group-based imputation.
   - Converting categorical features to string.
   - Label encoding for ordinal variables.

3. **Feature Engineering**
   - Creating new features: `TotalSF`, `Age`, `RemodAge`, `TotalBath`, `TotalPorchSF`
   - Log transformation of skewed features.

4. **Encoding**
   - One-hot encoding of categorical variables.

5. **Modeling**
   - Model: **XGBoost Regressor**
   - Cross-validation with RMSE metric.
   - Submission file generation.

---

## Model & Evaluation

- **Model**: XGBoost Regressor
- **Cross-Validation**: 5-fold
- **Evaluation Metric**: RMSE (Root Mean Squared Error)
- **Final Output**: CSV file (`house_price_submission.csv`) ready for Kaggle submission

---

## Installation

Make sure the following Python libraries are installed:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
