import pandas as pd
import nltk

from customerchurn.preprocessing import text_prepro


if __name__ == "__main__":
    nltk.download('omw-1.4')
    df = pd.read_csv('raw_data/concat.csv')
    df_sample= df.head(10)

    df_sample['review'] = df_sample['review'].apply(text_prepro)

    print(df_sample['review'])
