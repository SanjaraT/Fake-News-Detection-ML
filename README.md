📌 Project Overview

This project implements a complete end-to-end machine learning pipeline to classify news articles as Fake or Real using classical NLP and ML techniques.

The objective was to build a modular, professional-grade system with proper preprocessing, feature engineering, model comparison, and evaluation.

📂 Dataset

Dataset: Fake and Real News Dataset (Kaggle)
Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset


🧠 Methodology
--> Data Preparation

Combined Fake and True datasets

Assigned binary labels

Removed duplicate articles (after cleaning)

Performed stratified train-test split (70/30)

--> Text Preprocessing

Implemented in preprocess.py:

Lowercasing

HTML tag removal

Non-alphabetic character removal

Stopword removal

Lemmatization

-->  Models Trained

Naive Bayes (Baseline)

Logistic Regression

Linear SVM

Random Forest

XGBoost

--> Evaluation

Implemented in evaluation.py:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

All models were evaluated on the held-out test set.

📊 Results Summary

After removing duplicates and preventing leakage, the models achieved strong performance:

Naive Bayes: ~88–92%

Logistic Regression: ~94–97%

Linear SVM: ~96–98%

XGBoost: ~97–99%


⚠️ Limitations

The dataset contains source-specific writing patterns, making classification easier than real-world fake news detection.

Random train-test splitting may still allow stylistic similarities between train and test samples.

The model may not generalize to unseen news sources without domain adaptation.