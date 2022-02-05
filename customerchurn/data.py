import pandas as pd
from google.cloud import storage
from customerchurn.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH
from sklearn.model_selection import train_test_split


# def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
#     """method to get the training data (or a portion of it) from google cloud bucket"""
#     # Add Client() here
#     client = storage.Client()
#     path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
#     df = pd.read_csv(path, nrows=nrows)
#     return df

# def get_data(nrows=10000, path = './customerchurn/data/concat.csv', optimize=False, **kwargs):
#     """method to get the training data from local hard-drive"""
#     df = pd.read_csv( path, nrows=nrows)
#     return df


# def clean_data(df, test=False):
#     """Clean data: keep comments in english, specify the length of the review
#     convert stars to recommend/not recommended, keep only columns recommendation and review
#     and other feature engineering ideas"""
#     return df

# def holdout(df):

#     y = df["recommendation"]
#     X = df.drop("recommendation", axis=1)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#     return X_train, X_test, y_train, y_test

# if __name__ == '__main__':
#     df = get_data(path = './customerchurn/data/concat.csv')
