ARG FROM_IMAGE=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM ${FROM_IMAGE}

USER root

RUN apt-get update && apt-get install -y \
    git\
    vim\
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install -r requirements.txt

RUN python -m pip install 'zuko @ git+https://github.com/valsdav/zuko@master'
RUN python -m pip install 'mdmm @ git+https://github.com/the-moliver/mdmm@master'
