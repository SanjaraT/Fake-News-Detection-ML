from src.eda import load_and_preview
from src.preprocess import clean_text
from src.feature_extruction import add_text_features, tfidf_features
from sklearn.model_selection import train_test_split
from src.models import get_models
from src.evaluation import evaluate_model

df = load_and_preview("data/Fake.csv", "data/True.csv")

df['text'] = df['text'].apply(clean_text)

df = df.drop_duplicates(subset='text')
df = add_text_features(df)
X = df['text']
y = df['label']


#split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#tfidf
X_train_tfidf, X_test_tfidf, vectorizer = tfidf_features(X_train, X_test)

#training
models = get_models()

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)

    evaluate_model(model, X_test_tfidf, y_test, name)


