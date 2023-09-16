FROM python:3.9

RUN pip install tensorflow==2.12.0 tensorflow_datasets==4.9.2

COPY data-ingestion.py /
COPY predict-service.py /
COPY model-selection.py /
COPY multi-worker-distributed-training.py /