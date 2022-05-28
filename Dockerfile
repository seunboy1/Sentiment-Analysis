FROM python:3.8-slim

RUN pip install --upgrade pip
RUN apt-get update && apt-get -y upgrade \
                        gcc \
                        libc-dev

WORKDIR /src

COPY . .
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt