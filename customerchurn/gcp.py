import os

from google.cloud import storage
from termcolor import colored
from customerchurn.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION


def storage_upload(rm=False):
    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name = 'my_model.h5'
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename('my_model.h5')
    print(colored(f"=> my_model.h5 uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                  "green"))
    if rm:
        os.remove('my_model.h5')
