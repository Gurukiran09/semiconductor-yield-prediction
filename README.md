# Semiconductor Yield Prediction

Predicting semiconductor manufacturing yield with machine learning, feature engineering, and sensor-driven insights.

## 1. Project Overview

Semiconductor manufacturing involves monitoring a large number of process and sensor measurements to identify conditions that may lead to manufacturing failures.

This project applies machine learning techniques to semiconductor manufacturing sensor data to predict whether a production instance passes or fails.

The workflow covers data preprocessing, exploratory analysis, missing-value treatment, feature reduction, class-imbalance handling, model development, cross-validation, hyperparameter tuning, feature selection, and final model evaluation.

The main objective is to develop a classification model that can identify manufacturing failures while maintaining a practical balance between overall accuracy and minority-class detection.

---

## 2. Problem Statement

The dataset contains hundreds of sensor measurements collected during semiconductor manufacturing processes.

The target variable represents the manufacturing outcome:

- `-1` — Pass
- `1` — Fail

The dataset is highly imbalanced, with significantly more passing observations than failing observations.

This creates an important machine learning challenge because a model can achieve high overall accuracy while still performing poorly at detecting actual manufacturing failures.

Therefore, this project places particular emphasis on the detection of the minority `Fail` class.

---

## 3. Project Objectives

The main objectives of this project are:

- Explore and understand the semiconductor manufacturing dataset.
- Identify missing and potentially unreliable sensor measurements.
- Remove features that provide little or no useful information.
- Analyze relationships between sensor measurements and manufacturing outcomes.
- Prepare the data for machine learning.
- Address class imbalance using SMOTE.
- Develop and compare multiple classification models.
- Evaluate models using metrics beyond overall accuracy.
- Apply stratified cross-validation.
- Perform hyperparameter tuning using GridSearchCV.
- Identify the most influential sensor features.
- Evaluate models using different feature subsets.
- Select a final model based on its ability to detect manufacturing failures.
- Save the final trained model for potential future use.

---

## 4. Dataset

The dataset contains semiconductor manufacturing sensor measurements and a binary manufacturing outcome.

### Dataset Characteristics

- **Observations:** 1,567
- **Sensor features:** 590
- **Target variable:** Manufacturing outcome
- **Target classes:** Pass (`-1`) and Fail (`1`)
- **Pass observations:** 1,463
- **Fail observations:** 104

The target distribution is highly imbalanced:

- Pass: approximately 93.36%
- Fail: approximately 6.64%

The dataset also contains a substantial number of missing sensor measurements, making data preprocessing an important part of the project.

---

## 5. Methodology

### 5.1 Data Loading and Initial Exploration

The dataset was loaded using Pandas and examined to understand:

- Dataset dimensions
- Data types
- Sensor variables
- Target variable
- Missing values
- Basic statistical characteristics
- Class distribution

This initial exploration provided an overview of the structure and quality of the manufacturing data before model development.

### 5.2 Missing-Value Analysis

Missing-value analysis was performed across the sensor variables.

The dataset contained approximately 41,951 missing values distributed across 538 sensor features.

Features with more than 50% missing values were removed because such a high level of missing information could reduce their reliability and usefulness for prediction.

This resulted in the removal of 28 sensor features.

The remaining missing values were handled using median imputation.

### 5.3 Duplicate Record Analysis

The dataset was checked for duplicate observations.

No duplicate rows were identified, so no records needed to be removed at this stage.

### 5.4 Constant Feature Removal

Features with constant values across the dataset provide no useful information for classification.

A constant-feature analysis identified 116 sensor features with no meaningful variation.

These features were removed from the dataset.

After preprocessing, 446 sensor features remained for the initial machine learning experiments.

### 5.5 Exploratory Data Analysis

Exploratory analysis was performed to better understand the structure of the sensor data and identify potentially important relationships.

Correlation analysis was used to examine relationships between sensor measurements and the target variable.

Visualizations were also used to understand feature distributions and the imbalance between Pass and Fail observations.

### 5.6 Target Variable Analysis

The target variable was examined to understand the class distribution.

The dataset contains:

- 1,463 Pass observations
- 104 Fail observations

The strong class imbalance means that accuracy alone is not sufficient for evaluating model performance.

For this reason, particular attention was given to:

- Precision
- Recall
- F1-score
- Confusion matrix
- Fail-class detection

### 5.7 Feature and Target Preparation

The sensor measurements were separated from the target variable.

The target was converted into a binary classification problem where:

- `-1` represents Pass
- `1` represents Fail

The dataset was then divided into training and testing subsets using a stratified train-test split so that the class distribution was preserved.

### 5.8 Feature Scaling

Standardization was applied using `StandardScaler`.

Feature scaling was particularly important for models such as Logistic Regression and Support Vector Machines because these algorithms are sensitive to the relative scale of input variables.

The scaler was fitted using the training data and then applied to the relevant datasets.

### 5.9 Class Imbalance Handling

Because the Fail class represents only a small proportion of the dataset, Synthetic Minority Oversampling Technique (SMOTE) was applied to the training data.

Before SMOTE:

- Pass: 1,170
- Fail: 83

After SMOTE:

- Pass: 1,170
- Fail: 1,170

SMOTE was applied only to the training data so that the original distribution of the test data remained unchanged.

During cross-validation and hyperparameter tuning, SMOTE was incorporated into machine learning pipelines to help prevent data leakage.

---

## 6. Machine Learning Models

Three classification algorithms were developed and compared.

### 6.1 Logistic Regression

Logistic Regression was used as a baseline classification model.

It provides a simple and interpretable benchmark for evaluating whether more complex models provide meaningful improvements.

### 6.2 Random Forest

Random Forest was evaluated because it can model nonlinear relationships and interactions between sensor measurements.

It also provides feature-importance information that can be useful for understanding which sensors contribute most to prediction.

### 6.3 Support Vector Machine

Support Vector Machine (SVM) was evaluated as a classification approach capable of handling high-dimensional feature spaces.

The SVM model produced stronger minority-class performance compared with the other baseline models and was therefore selected for further optimization and feature-selection analysis.

---

## 7. Model Evaluation

Model performance was evaluated using multiple classification metrics rather than relying solely on accuracy.

The evaluation included:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

Particular attention was given to the `Fail` class because correctly identifying manufacturing failures is more important than simply maximizing overall accuracy.

---

## 8. Cross-Validation

Stratified 5-fold cross-validation was used to obtain a more reliable estimate of model performance.

Stratification ensured that each fold maintained a similar proportion of Pass and Fail observations.

SMOTE was incorporated within the cross-validation pipeline so that oversampling was performed independently within each training fold.

This approach helps reduce the risk of information leakage between training and validation data.

---

## 9. Hyperparameter Tuning

GridSearchCV was used to identify suitable hyperparameter combinations for the machine learning models.

The models were optimized using the F1-score of the Fail class as the primary scoring metric.

This choice reflects the project's focus on identifying manufacturing failures rather than optimizing overall accuracy alone.

### Logistic Regression

The tuning process evaluated different values of:

- Regularization strength (`C`)
- Solver

### Random Forest

The tuning process evaluated combinations of:

- Number of trees
- Maximum tree depth
- Minimum samples required for splitting
- Minimum samples required at leaf nodes

### Support Vector Machine

The SVM tuning process evaluated:

- Regularization parameter (`C`)
- Kernel
- Gamma

---

## 10. Feature Selection

Feature selection was performed using a linear SVM.

The absolute values of the SVM coefficients were used to rank the sensor features according to their contribution to the classification decision.

The highest-ranked features were then evaluated to determine whether a smaller subset of sensors could provide effective failure detection.

The top 20 selected sensor features were:

1. `Sensor_60`
2. `Sensor_217`
3. `Sensor_189`
4. `Sensor_125`
5. `Sensor_26`
6. `Sensor_133`
7. `Sensor_62`
8. `Sensor_426`
9. `Sensor_355`
10. `Sensor_278`
11. `Sensor_27`
12. `Sensor_57`
13. `Sensor_42`
14. `Sensor_564`
15. `Sensor_306`
16. `Sensor_65`
17. `Sensor_169`
18. `Sensor_334`
19. `Sensor_416`
20. `Sensor_216`

---

## 11. Feature Subset Evaluation

Several feature configurations were compared:

- Top 20 features
- Top 50 features
- Top 100 features
- All 446 remaining features

The purpose was to determine whether a smaller set of influential sensors could achieve competitive performance while reducing the dimensionality of the model.

The full 446-feature configuration achieved the highest overall accuracy.

However, the Top 20 feature configuration achieved the strongest Recall and F1-score for the Fail class among the evaluated feature configurations.

---

## 12. Model Comparison

The initial model comparison demonstrated that overall accuracy alone can be misleading for this dataset.

### Logistic Regression

Logistic Regression provided a baseline but performed poorly at identifying failures.

The confusion matrix showed that only 3 Fail observations were correctly detected, while 18 actual Fail observations were classified as Pass.

### Random Forest

Random Forest achieved high overall accuracy but failed to correctly identify the Fail observations in the evaluated test set.

All 21 actual Fail observations were classified as Pass.

This demonstrates why accuracy alone is not an appropriate metric for this highly imbalanced problem.

### Support Vector Machine

SVM provided better performance for the minority Fail class and was therefore selected for further optimization and feature-selection analysis.

---

## 13. Final Model Selection

The final model was selected based on the project's primary objective of identifying manufacturing failures.

The selected approach was:

**Tuned Linear Support Vector Machine using the Top 20 selected sensor features.**

The Top 20 feature model correctly identified 13 of the 21 Fail observations in the test set.

Its Fail-class performance included:

- **Recall:** 0.62
- **F1-score:** 0.22

Although using all 446 features produced higher overall accuracy, the Top 20 configuration provided stronger failure detection.

Therefore, the Top 20 feature SVM was selected because the project prioritizes identifying failures rather than maximizing raw accuracy.

---

## 14. Key Findings

Several important findings emerged from the analysis:

1. The dataset is strongly imbalanced toward successful manufacturing outcomes.

2. Missing-value treatment is essential because a large number of sensor measurements contain missing values.

3. Removing highly incomplete and constant features reduces unnecessary dimensionality.

4. SMOTE improves the representation of the minority Fail class during model training.

5. Accuracy alone can give a misleading impression of model performance on imbalanced manufacturing data.

6. Random Forest achieved strong overall accuracy but failed to identify actual failures in the evaluated test set.

7. SVM provided better minority-class performance and was therefore selected for further analysis.

8. Feature selection using SVM coefficients identified a small subset of influential sensor variables.

9. The full feature set produced the highest overall accuracy, while the Top 20 feature set provided better Fail-class Recall and F1-score.

10. The final model prioritizes failure detection over maximum overall accuracy.

---

## 15. Conclusion

This project demonstrates an end-to-end machine learning workflow for semiconductor manufacturing yield prediction.

The analysis showed that handling missing values, removing uninformative features, addressing class imbalance, and selecting appropriate evaluation metrics are critical when working with highly imbalanced manufacturing data.

Multiple machine learning models were evaluated, followed by cross-validation, hyperparameter tuning, and feature selection.

The final approach uses a tuned linear SVM with the 20 most influential sensor features.

The model achieved a Fail-class Recall of 0.62 on the evaluated test set, correctly identifying 13 of 21 Fail observations.

The results highlight the importance of selecting a model according to the actual business objective. In this case, detecting manufacturing failures is more valuable than simply maximizing overall classification accuracy.

---
