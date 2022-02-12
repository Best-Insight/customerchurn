import os
import pandas as pd

from google.cloud import storage

from customerchurn.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH

# from sklearn.model_selection import train_test_split
# from tensorflow.data.Dataset import from_generator
import tensorflow as tf


DATASET_SIZE = 8_635_403

def data_generator(folder):
    """
    get data frome given folder
    Returns:
        iterate data
    """
    files = [file for file in os.listdir(folder)]
    for file in files:
        file_path = os.path.join(folder, file)
        # read one file
        rows = pd.read_csv(file_path)
        for i in range(512):
            x = rows.review.iloc[i]
            if rows.star.iloc[i] >= 4:
                y = 1
            else:
                y = 0
            yield x, y


def get_dataset(folder = '../raw_data/yelp', optimize=False, **kwargs):
    """method to get the training data from local hard-drive"""
    data_set = tf.data.Dataset.from_generator(
        lambda: data_generator(folder),
        output_types=(tf.string, tf.bool)).batch(256)
    return data_set.shuffle(buffer_size=2048)


def split_dataset(dataset):
    train_size = int(0.7 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, val_dataset, test_dataset


def star2recommend(df, good=4):
    df['recommendation'] = df['star'].apply(
        lambda x: 'Recommended' if x >= good else 'Not Recommended')
    return df


# def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
#     """method to get the training data (or a portion of it) from google cloud bucket"""
#     # Add Client() here
#     client = storage.Client()
#     path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
#     df = pd.read_csv(path, nrows=nrows)
#     return df

def get_data(nrows=10000, path = './customerchurn/data/rwa_en.csv', optimize=False, **kwargs):
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
