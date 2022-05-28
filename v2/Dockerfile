FROM apache/airflow:2.2.1
USER root

RUN pip install --upgrade pip
RUN apt-get update && apt-get -y upgrade

ENV MODEL_DIR=/src/Models
ENV MODEL_FILE=model.pkl
ENV METADATA_FILE=data/IMDB_Dataset.csv

WORKDIR /src

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt


# docker build -t biosciences:v1 -f Dockerfile .

