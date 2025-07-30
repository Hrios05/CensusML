# Census Income Classification

## 📌 Project Overview

This project follows the complete machine learning life cycle to solve a real-world classification problem using the **1994 Census Income Dataset**. The objective is to **predict whether an individual's income exceeds \$50K/year** based on demographic and work-related attributes.

This notebook includes:
- Exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering and selection
- Model training and evaluation using:
  - Logistic Regression
  - Random Forest
  - Neural Networks

---

## 📁 Dataset Used

**File**: `censusData.csv`  
**Source**: UCI Machine Learning Repository (modified for educational use)

### 🎯 Problem Definition
- **Type**: Supervised Learning
- **Category**: Binary Classification
- **Target Variable**: `income_binary` (0 = ≤50K, 1 = >50K)
- **Features**:
  - Continuous: `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
  - Categorical (one-hot encoded): `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex_selfID`, `native-country`

---

## ⚙️ ML Workflow

### 🧼 Data Preprocessing
- Imputed missing values using column means (for `age` and `hours-per-week`) or dropped incomplete rows
- Encoded the target variable using `LabelEncoder`
- Applied **one-hot encoding** to all categorical features
- Standardized features using `StandardScaler`

### 🔍 Feature Selection
- Used **Random Forest** to extract the top 50 most important features to reduce noise

---

## 🤖 Models Implemented

### 1. Neural Network (TensorFlow/Keras)
- 3 hidden layers with ReLU activation
- **Batch Normalization** and **Dropout** used for regularization
- Optimizers: Tested with both `SGD` and `Adam`
- Trained for 115 epochs with early insight logging every 5 epochs

**Performance**:
- Accuracy on test set: **~83.9%**
- Good generalization with training vs. validation accuracy curves closely aligned

### 2. Logistic Regression (Baseline Model)
- Implemented with `sklearn.linear_model.LogisticRegression`
- Achieved comparable performance to neural network

**Performance**:
- Accuracy on test set: **~84.2%**
- Simple, fast to train, and interpretable

---

## 📊 Evaluation

- Metric used: **Accuracy**
- Also evaluated predicted probabilities vs. actual classes
- Plotted training/validation loss and accuracy over epochs
- Compared model complexity vs. performance for fairness

---

## ✅ Key Takeaways

- **Model Selection** matters: simple models like Logistic Regression can match the performance of more complex ones for some datasets.
- **Feature Engineering and Preprocessing** are critical: handling missing data and one-hot encoding were essential to model success.
- **Bias/Fairness Consideration**: This problem sheds light on how income inequality correlates with social and demographic features, which can have ethical implications in real-world deployments.

---

## 🛠️ Technologies Used

- Python 3  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib, Seaborn  

---

## 📚 Author Notes

This notebook was completed as **Lab 8** of the Machine Learning curriculum. The lab required defining a predictive task, developing a complete ML pipeline, and reflecting on model performance and fairness. The Census dataset was chosen for its rich demographic insights and relevance to social equity research.
