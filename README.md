# Project 2 - Fraud Detection Using Machine Learning

## 📌 Project Overview

This project develops a machine learning pipeline for identifying potentially fraudulent transactions in an e-commerce sales dataset.

The objective is to build and evaluate classification models capable of distinguishing potentially fraudulent transactions from normal transactions while addressing class imbalance and optimizing model performance.

The project follows a data science workflow including data preprocessing, feature engineering, class-imbalance handling, model training, hyperparameter tuning, evaluation, and classification threshold optimization.

## 🛠️ Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Jupyter Notebook

## 📊 Dataset

The project uses an e-commerce sales dataset containing **1,200 orders and 14 columns**.

The dataset includes information such as:

* OrderID
* Date
* CustomerID
* Product
* Quantity
* UnitPrice
* ShippingAddress
* PaymentMethod
* OrderStatus
* TrackingNumber
* ItemsInCart
* CouponCode
* ReferralSource
* TotalPrice

The original dataset does not contain a verified real-world fraud label. Therefore, the fraud target used in this project was constructed using the defined fraud criteria.

## 🎯 Project Objectives

* Prepare transaction data for machine learning
* Engineer relevant features
* Handle class imbalance using SMOTE
* Build classification models
* Perform hyperparameter tuning
* Compare model performance
* Evaluate fraud detection using appropriate classification metrics
* Analyze Precision-Recall performance
* Optimize the classification threshold
* Select the most suitable final model

## 🔍 Methodology

The project workflow consists of:

1. Data loading and inspection
2. Exploratory data analysis
3. Feature engineering
4. Target preparation
5. Train-test split
6. Numerical and categorical feature preprocessing
7. Class imbalance analysis
8. SMOTE-based oversampling
9. Logistic Regression training
10. Logistic Regression hyperparameter tuning
11. Random Forest training
12. Random Forest hyperparameter tuning
13. Model evaluation
14. Precision-Recall analysis
15. Classification threshold analysis
16. Final model selection

## 🤖 Models Used

### Logistic Regression

Logistic Regression was used as a baseline classification model.

Best hyperparameter:

```text
C = 0.01
```

Test ROC-AUC:

```text
0.9792
```

Fraud Recall:

```text
91.67%
```

Fraud Precision:

```text
31.43%
```

### Random Forest

Random Forest was used as a nonlinear ensemble classification model.

Best hyperparameters:

```text
n_estimators = 100
max_depth = None
min_samples_split = 5
```

Best Cross-Validation ROC-AUC:

```text
0.9841
```

Test ROC-AUC:

```text
0.9898
```

Average Precision:

```text
0.8897
```

## 📊 Model Comparison

| Model                | ROC-AUC | Average Precision | Precision | Recall | F1-Score |
| -------------------- | ------: | ----------------: | --------: | -----: | -------: |
| Logistic Regression  |  0.9792 |            0.7708 |    0.3143 | 0.9167 |   0.4681 |
| Random Forest (0.50) |  0.9898 |            0.8897 |    1.0000 | 0.5833 |   0.7368 |
| Random Forest (0.35) |  0.9898 |            0.8897 |    0.8333 | 0.8333 |   0.8333 |

## 🎯 Threshold Optimization

The default Random Forest classification threshold of **0.50** produced very high precision but lower fraud recall.

Different thresholds were evaluated to find a better balance between precision and recall.

The selected threshold was:

```text
0.35
```

At this threshold:

* Precision = **83.33%**
* Recall = **83.33%**
* F1-Score = **83.33%**

This provided a better balance between detecting fraudulent transactions and minimizing false fraud alerts.

## 🏆 Final Model

The final selected model is:

**Random Forest with a classification threshold of 0.35**

### Final Test Results

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          | 98.33% |
| ROC-AUC           | 0.9898 |
| Average Precision | 0.8897 |
| Fraud Precision   | 83.33% |
| Fraud Recall      | 83.33% |
| Fraud F1-Score    | 83.33% |

### Final Confusion Matrix

```text
[[226   2]
 [  2  10]]
```

This represents:

* True Negatives: **226**
* False Positives: **2**
* False Negatives: **2**
* True Positives: **10**

The final model correctly identified 10 out of 12 potentially fraudulent transactions while generating only 2 false-positive predictions.

## 💡 Key Findings

* Random Forest achieved a higher test ROC-AUC than Logistic Regression.
* Random Forest also achieved substantially higher Average Precision.
* Logistic Regression provided higher fraud recall at the default threshold but produced many false positives.
* Random Forest at the default threshold achieved perfect precision but missed several fraud cases.
* Reducing the Random Forest threshold from 0.50 to 0.35 improved fraud recall from **58.33% to 83.33%**.
* The 0.35 threshold produced an equal precision and recall of **83.33%**.
* Based on the evaluated metrics, Random Forest with a threshold of 0.35 was selected as the final model.

## ⚠️ Limitations

* The test set contained only **12 positive cases**, so fraud-related metrics may be sensitive to a small number of predictions.
* The dataset does not contain a verified real-world fraud label.
* The fraud target was constructed according to the project's defined criteria.
* Therefore, the model should be interpreted as detecting transactions matching the project's fraud criteria rather than confirmed real-world fraud.
* The selected threshold of 0.35 was optimized using the available evaluation setup and should not be considered universally optimal.

## 🚀 Future Improvements

Future work could include:

* Using a larger dataset
* Using verified fraud labels
* Applying time-based train-validation-test splitting
* Optimizing the threshold using a separate validation set
* Testing additional machine learning algorithms
* Applying cost-sensitive learning
* Evaluating the financial cost of false positives and false negatives
* Performing model interpretability using feature importance or SHAP
* Monitoring model performance on new transactions

## 📁 Project Files

* `Project2_Fraud_Detection.ipynb` - Complete machine learning workflow, model training, evaluation, and threshold analysis.
* `README.md` - Project documentation.

## 👨‍💻 Author

**Mehdi Raza**

BS Computer Science Student
