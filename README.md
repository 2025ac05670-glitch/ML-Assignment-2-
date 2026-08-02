# Machine Learning Assignment 2 - Adult Income Classification

## a. Problem Statement
The objective of this project is to predict whether a person's income is greater than 50K per year or less than/equal to 50K per year using census-related demographic and employment attributes. This is a binary classification problem.

## b. Dataset Description
Dataset Name: Adult Income / Census Income Dataset

Source: UCI Machine Learning Repository or Kaggle Adult Census Income Dataset

The dataset contains details such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, native country, and income.

Target variable: income
Classes: <=50K and >50K

The downloaded dataset used in this assignment contains 32,561 instances and 15 columns, satisfying the assignment requirement of minimum 500 instances and minimum 12 features.

## c. GitHub Repository Link
https://github.com/your-username/ML-Assignment-2

## d. Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier
5. Random Forest Classifier

## Evaluation Metrics
Accuracy, AUC Score, Precision, Recall, F1 Score, Matthews Correlation Coefficient

## Model Comparison Table
Paste the table from metrics_results.csv here.

## Model Performance Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good baseline model. Performs well when the feature-target relationship is mostly linear. |
| Decision Tree | Easy to understand but may overfit. |
| KNN | Gives reasonable performance but can be slower for larger datasets. |
| Naive Bayes | Fast and simple. Performance depends on independence assumption between features. |
| Random Forest | Usually strong because multiple trees reduce overfitting and improve generalization. |
| Overall Winner | Select the model with the highest F1 and MCC score from the metrics table. |

## Streamlit App Link
https://your-app-name.streamlit.app

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files Included
app.py, requirements.txt, README.md, test_data.csv, metrics_results.csv, models folder, and notebook file.
