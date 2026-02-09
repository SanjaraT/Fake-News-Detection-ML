from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models():
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "XGBoost": XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False
        )
    }
    return models
