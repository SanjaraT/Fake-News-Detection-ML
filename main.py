from src.eda import load_and_preview
from src.preprocess import clean_text

df = load_and_preview("data/Fake.csv", "data/True.csv")

df['text'] = df['text'].apply(clean_text)
