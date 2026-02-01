import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preview(fake_path, true_path):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake['label'] = 0
    true['label'] = 1

    df = pd.concat([fake, true], axis=0).reset_index(drop=True)

    print("Dataset shape:", df.shape)
    print(df.head())
    print("\nClass distribution:\n", df['label'].value_counts())

    return df