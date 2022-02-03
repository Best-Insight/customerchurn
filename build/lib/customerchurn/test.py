import pandas as pd

from customerchurn.preprocessing import text_prepro


if __name__ == "__main__":
    df = pd.read_csv('../raw_data/concat.csv')
    df_sample= df.head(10)

    df_sample['reviews'] = df_sample['reviews'].apply(text_prepro)

    print(df_sample['reviews'])
