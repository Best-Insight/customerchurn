# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* customerchurn/*.py

black:
	@black scripts/* customerchurn/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr customerchurn-*.dist-info
	@rm -fr customerchurn.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ----------------------------------
# GCP COMPONENT
# ----------------------------------

# path of the file to upload to gcp (the path of the file should be absolute or should match the directory where the make command is run)
LOCAL_PATH_MODEL=models/Financial_Sevices_model

LOCAL_PATH_DATA=raw_data/yelp_heading_split/yelp_score_5
LOCAL_PATH_DATA_FOLDER=raw_data/yelp_heading_split

# project id
PROJECT_ID=lewagon-alen-28296

# bucket name
BUCKET_NAME=wagon-data-709-melbourne-customerchurn-alen

# bucket directory in which to store the uploaded file (we choose to name this data as a convention)
BUCKET_FOLDER_MODEL=models
BUCKET_FOLDER_DATA=data/yelp_heading_split

# name for the uploaded file inside the bucket folder (here we choose to keep the name of the uploaded file)
# BUCKET_FILE_NAME=another_file_name_if_I_so_desire.csv
BUCKET_FILE_NAME_MODEL=$(shell basename ${LOCAL_PATH_MODEL})
BUCKET_FILE_NAME_DATA=$(shell basename ${LOCAL_PATH_DATA})
BUCKET_FILE_NAME_DATA_FOLDER=$(shell basename ${LOCAL_PATH_DATA_FOLDER})

REGION=asia-southeast1

set_project:
	-@gcloud config set project ${PROJECT_ID}

create_bucket:
	-@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}


upload_model:
	-@gsutil -m cp -r ${BUCKET_FOLDER_MODEL} gs://${BUCKET_NAME}/${BUCKET_FOLDER_MODEL}/${BUCKET_FILE_NAME_MODEL}


upload_data:
	-@gsutil cp ${LOCAL_PATH_DATA} gs://${BUCKET_NAME}/${BUCKET_FOLDER_DATA}/${BUCKET_FILE_NAME_DATA}

upload_data_folder:
	-@gsutil -m cp -r ${LOCAL_PATH_DATA_FOLDER} gs://${BUCKET_NAME}/${BUCKET_FOLDER_DATA}/${BUCKET_FILE_NAME_DATA_FOLDER}

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

# BUCKET_NAME=XXX

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

##### Machine configuration - - - - - - - - - - - - - - - -

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=2.8

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=customerchurn
FILENAME=trainer

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=customerchurn_yelp_score_5_combined_$(shell date +'%Y%m%d_%H%M%S')


run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--scale-tier custom \
		--master-machine-type a2-highgpu-1g \
		--master-accelerator count=1,type=nvidia-tesla-a100 \
		--region ${REGION} \
		--stream-logs


##### Prediction API - - - - - - - - - - - - - - - - - - - - - - - - -

run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload
