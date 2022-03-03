### MLFLOW configuration - - - - - - - - - - - - - - - - - - -
MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[AUS] [MEL] [NLP709] CUSTOMERCHURN"

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'models/sentiment_model1'

DATASET_SIZE = 8_635_403

#AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"


### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = "wagon-data-709-melbourne-customerchurn-alen"

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/yelp_heading_split/yelp_score_5'


##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

BUCKET_TRAIN_MODEL_PATH = 'models'

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'customerchurn'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v2'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -
