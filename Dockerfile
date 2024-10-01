FROM ubuntu:22.04

# Install dependencies

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python-is-python3
RUN apt-get install -y python3.9-distutils
RUN echo "alias pip='python3 -m pip'" >> ~/.bashrc
RUN echo "alias pip3='python3 -m pip'" >> ~/.bashrc
RUN echo "alias python3='python3.9'" >> ~/.bashrc
RUN echo "alias python='python3.9'" >> ~/.bashrc

COPY requirements.txt requirements.txt
RUN python3.9 -m pip install -r requirements.txt

WORKDIR /AnimalAI