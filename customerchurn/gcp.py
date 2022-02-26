import os

from google.cloud import storage
from termcolor import colored
from customerchurn.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION

# _file1 = 'my_model/assets/vocab.txt'
# _file2 = 'my_model/variables/variables.data-00000-of-00001'
# _file3 = 'my_model/variables/variables.index'
# _file4 = 'my_model/keras_metadata.pb'
# _file5 = 'my_model/saved_model.pb'




def storage_upload_folder(rm=False, folder_name='my_model', gcp_folder='models'):
    file_path = []
    for parent, dirnames, filenames in os.walk(folder_name):
        for filename in filenames:
            file_path.append(parent + '/' + filename)
    client = storage.Client().bucket(BUCKET_NAME)
    for local_model_name in file_path:
        storage_location = f"{gcp_folder}/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
        blob = client.blob(storage_location)
        blob.upload_from_filename(local_model_name)
        # print(colored(f"=> my_model uploaded to bucket {BUCKET_NAME} inside {storage_location}",
        #             "green"))
        if rm:
            os.remove(local_model_name)


def storage_upload_file(rm=False,
                        file_name='my_model_history.json',
                        gcp_folder='history'):
    client = storage.Client().bucket(BUCKET_NAME)
    local_model_name = f'{gcp_folder}/{file_name}'
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(file_name)
    # print(colored(f"=> my_model_history.json uploaded to bucket {BUCKET_NAME} inside {storage_location}",
    #               "green"))
    if rm:
        os.remove(file_name)
