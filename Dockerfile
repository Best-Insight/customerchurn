From python:3.8.12-buster



COPY customerchurn /customerchurn
COPY requirements.txt /requirements.txt
COPY api /api
COPY models/Financial_Services_model /models/Financial_Services_model

RUN pip install -U pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
