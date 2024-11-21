FROM python:3

RUN apt-get update -y \
 && apt-get install -y \
        infernal \
 && rm -r /var/lib/apt/lists/*

RUN pip install RNApolis
