FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-tk \
    python-opencv

RUN mkdir /musictracker
WORKDIR /musictracker

ADD requirements.txt /musictracker
ADD train.py /musictracker
ADD classify.py /musictracker
ADD classify_cam.py /musictracker
COPY dataset /musictracker/dataset/

RUN pip3 install -r requirements.txt
