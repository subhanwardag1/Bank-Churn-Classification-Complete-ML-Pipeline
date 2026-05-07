# Bank Churn Classification - Machine Learning Project

## 📋 Project Overview

This is a professional machine learning pipeline that implements binary classification to predict customer churn in a banking context. The project uses two classification algorithms (**Logistic Regression** and **K-Nearest Neighbors**) to identify customers likely to leave the bank, enabling proactive customer retention strategies.

**Dataset:** `bank_churn_modelling.csv` - Contains customer banking information with churn status as the target variable.

---

## 🎯 Objectives

1. Load and explore the bank churn dataset
2. Perform comprehensive exploratory data analysis (EDA)
3. Preprocess data (encoding, scaling, feature engineering)
4. Build and train Logistic Regression model
5. Build and train K-Nearest Neighbors model
6. Evaluate both models using multiple metrics
7. Perform hyperparameter tuning for both models
8. Visualize model performance and results
9. Compare model performances
10. Generate final predictions and model insights

---

## 📁 Project Structure

```
ML_Project/
├── bank_churn_modelling.csv    # Raw dataset
├── ml2.ipynb                    # Complete ML pipeline (main notebook)
└── README.md                    # This file
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Project

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook ml2.ipynb
   ```

2. **Execute cells sequentially** from top to bottom to see the complete pipeline

3. **View outputs** including statistics, visualizations, and model comparisons

---

## 📊 Pipeline Implementation

### Task 1: Data Loading
- Load the CSV file into a pandas DataFrame
- Display dataset shape: 10,000 rows × 14 columns
- Show data types and missing values
- Print first few records for inspection

### Task 2: Exploratory Data Analysis (EDA)
- Statistical summary (count, mean, std, min, max)
- Identify categorical and numerical columns
- Analyze target variable distribution (Churn vs. No Churn)
- Visualize class imbalance

### Task 3: Data Visualization - Distributions
- Create histograms for numerical features
- Display distributions across multiple subplots
- Identify skewed and normally distributed features
- Analyze feature ranges and outliers

### Task 4: Data Visualization - Correlations
- Generate correlation matrix heatmap
- Identify features highly correlated with churn
- Spot multicollinearity issues
- Visualize relationships between features

### Task 5: Data Preprocessing - Encoding
- **Categorical Encoding:**
  - Apply Label Encoding to 'Gender' column
  - Apply One-Hot Encoding to 'Geography' column
- **Target Variable Encoding:**
  - Encode 'Exited' (churn) as binary (0/1)
- **Drop Irrelevant Columns:**
  - Remove 'RowNumber', 'CustomerId', 'Surname' (non-predictive)

### Task 6: Data Preprocessing - Feature Scaling
- Apply **StandardScaler** to normalize features
- Ensure all features have mean=0 and std=1
- Necessary for distance-based algorithms (KNN)
- Improve Logistic Regression convergence

### Task 7: Train-Test Split
- Split data: 80% training, 20% testing
- Maintain randomization (random_state=42) for reproducibility
- Stratified split to preserve class distribution
- Separate features (X) and target (y)

### Task 8: Logistic Regression - Model Building & Training
- **Model Configuration:**
  - Algorithm: Logistic Regression with L2 regularization
  - Max iterations: 1000 (sufficient for convergence)
  - Random state: 42 (reproducibility)
- **Training:** Fit on training data (X_train, y_train)
- **Predictions:** Generate predictions on train and test sets
- **Evaluation Metrics:**
  - Training Accuracy
  - Testing Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-Score

### Task 9: K-Nearest Neighbors - Model Building & Training
- **Model Configuration:**
  - Algorithm: KNN with K=5 neighbors (default)
  - Distance metric: Euclidean
- **Training:** Fit on training data
- **Predictions:** Generate predictions on train and test sets
- **Evaluation Metrics:**
  - Training Accuracy
  - Testing Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-Score

### Task 10: Hyperparameter Tuning & Model Comparison
- **Logistic Regression Tuning:**
  - Test C values: [0.001, 0.01, 0.1, 1, 10, 100]
  - Identify optimal C value
  - Compare with baseline model
  
- **KNN Tuning:**
  - Test K values: [1, 3, 5, 7, 9, 11, 13, 15]
  - Track accuracy for each K
  - Identify optimal K value
  
- **Model Comparison:**
  - Compare final tuned models
  - Display accuracy scores side-by-side
  - Identify best performing model
  - Generate feature importance analysis

---

## 📈 Key Results

### Model Performance Summary

| Metric | Logistic Regression | K-Nearest Neighbors |
|--------|-------------------|-------------------|
| Training Accuracy | High | High |
| Testing Accuracy | Optimized | Optimized |
| Precision | Calculated | Calculated |
| Recall | Calculated | Calculated |
| F1-Score | Calculated | Calculated |

*Actual values displayed in notebook outputs*

### Key Insights

- **Feature Importance:** Top contributing features identified through correlation analysis
- **Class Distribution:** Analysis of churn vs. non-churn customers
- **Model Comparison:** Best model identified based on test accuracy and F1-score
- **Optimal Hyperparameters:** 
  - LR: Optimal C value determined through grid search
  - KNN: Optimal K value identified for best accuracy

---

## 🛠️ Technologies & Libraries

| Technology | Purpose |
|-----------|---------|
| **Python** | Programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Machine learning algorithms & metrics |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical data visualization |
| **Jupyter Notebook** | Interactive development environment |

---

## 📚 Methodology

### Machine Learning Workflow

```
1. Data Loading
   ↓
2. Exploratory Data Analysis (EDA)
   ↓
3. Data Visualization
   ↓
4. Data Preprocessing
   - Encoding categorical variables
   - Feature scaling
   ↓
5. Train-Test Split
   ↓
6. Model Building (Two Approaches)
   ├─→ Logistic Regression
   └─→ K-Nearest Neighbors
   ↓
7. Initial Model Evaluation
   ↓
8. Hyperparameter Tuning
   ├─→ LR: Test C values
   └─→ KNN: Test K values
   ↓
9. Final Model Comparison
   ↓
10. Conclusions & Recommendations
```

### Evaluation Metrics Used

1. **Accuracy:** Overall correctness of predictions
2. **Precision:** True positives among predicted positives (minimize false positives)
3. **Recall:** True positives among actual positives (minimize false negatives)
4. **F1-Score:** Harmonic mean of precision and recall (balanced metric)
5. **Confusion Matrix:** Visual representation of classification performance
6. **Classification Report:** Comprehensive metric summary

---

## 🔍 Code Quality Features

- **Clear Comments:** Each section is well-documented
- **Modular Structure:** Organized into logical tasks
- **Professional Naming:** Descriptive variable and function names
- **Error Handling:** Warnings suppressed for clean output
- **Reproducibility:** Random seeds set for consistent results
- **Visualization Style:** Consistent and professional plot formatting
- **Output Organization:** Structured print statements with separators

---

## 💡 Usage Examples

### Running the Complete Pipeline
```python
# Execute all cells in order to run the complete pipeline
# Each cell builds upon the previous one
```

### Interpreting Confusion Matrix
- **True Negatives (TN):** Correctly predicted no-churn
- **False Positives (FP):** Incorrectly predicted churn
- **False Negatives (FN):** Incorrectly predicted no-churn
- **True Positives (TP):** Correctly predicted churn

### Understanding Hyperparameter Tuning Results
- Review accuracy curves for both models
- Identify the K value where KNN accuracy plateaus
- Identify the C value that maximizes LR accuracy
- Compare baseline vs. tuned performance

---

## 🎓 Learning Outcomes

After completing this project, you will understand:

✅ Complete ML pipeline from data loading to model deployment  
✅ Difference between Logistic Regression and KNN algorithms  
✅ Importance of data preprocessing and feature scaling  
✅ Hyperparameter tuning and model optimization  
✅ Evaluation metrics for classification problems  
✅ How to handle categorical and numerical features  
✅ Class imbalance awareness  
✅ Train-test split strategy  
✅ Model comparison and selection  

---



