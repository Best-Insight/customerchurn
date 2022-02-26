import os
import pandas as pd

from google.cloud import storage

from customerchurn.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, DATASET_SIZE

# from sklearn.model_selection import train_test_split
# from tensorflow.data.Dataset import from_generator
import tensorflow as tf


def data_generator():
    """
    get data frome given folder
    Returns:
        iterate data
    """
    client = storage.Client()
    for i in range(1, 19):
        path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}/yelp_review{i}"
        rows = pd.read_csv(path)
        for i in range(len(rows)):
            x = rows.review.iloc[i]
            if rows.star.iloc[i] >= 4:
                y = 1
            else:
                y = 0
            yield x, y


def get_dataset_from_gcp(optimize=False, **kwargs):
    """method to get the training data from local hard-drive"""
    data_set = tf.data.Dataset.from_generator(
        lambda: data_generator(),
        output_types=(tf.string, tf.bool)).batch(128)
    return data_set.shuffle(buffer_size=4096)


def split_dataset(dataset):
    train_size = int(0.7 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, val_dataset, test_dataset


def star2recommend(df, good=4):
    if 'star' in df.columns:
        df['recommendation'] = df['star'].apply(
            lambda x: 'Recommended' if x >= good else 'Not Recommended')
    return df


def get_data_from_gcp(n_rows=None, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
    df = pd.read_csv(path, nrows=n_rows)
    return df


def get_data_from_gcp_folder(n_rows=None, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    df_all = pd.DataFrame()
    for i in range(1, 19):
        path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}/yelp_review{i}"
        df = pd.read_csv(path, nrows=n_rows)
        df = star2recommend(df, good=4)
        df = df[['recommendation', 'review']]
        df_all = pd.concat([df_all, df])
    return df_all


def get_data(nrows=10000, path = './customerchurn/data/raw_en.csv', optimize=False, **kwargs):
    """method to get the training data from local hard-drive"""
    df = pd.read_csv( path, nrows=nrows)
    df = star2recommend(df)
    return df


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
