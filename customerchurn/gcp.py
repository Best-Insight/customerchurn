import os

from google.cloud import storage
from termcolor import colored
from customerchurn.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION

# _file1 = 'my_model/assets/vocab.txt'
# _file2 = 'my_model/variables/variables.data-00000-of-00001'
# _file3 = 'my_model/variables/variables.index'
# _file4 = 'my_model/keras_metadata.pb'
# _file5 = 'my_model/saved_model.pb'

file_path = []
for parent, dirnames, filenames in os.walk('my_model'):
    for filename in filenames:
        file_path.append(parent + '/' + filename)


def storage_upload(rm=False):
    client = storage.Client().bucket(BUCKET_NAME)
    for local_model_name in file_path:
        storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
        blob = client.blob(storage_location)
        blob.upload_from_filename(local_model_name)
        print(colored(f"=> my_model uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                    "green"))
        if rm:
            os.remove(local_model_name)

def history_upload(rm=False):
    client = storage.Client().bucket(BUCKET_NAME)
    local_model_name = 'history/my_model_history.json'
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename('my_model_history.json')
    print(colored(f"=> my_model_history.json uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                  "green"))
    if rm:
        os.remove('my_model_history.json')
