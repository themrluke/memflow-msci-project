ARG FROM_IMAGE=pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM ${FROM_IMAGE}


ADD requirements.txt /tmp/
RUN python -m pip install -r /tmp/requirements.txt

RUN python -m pip install 'zuko @ git+https://github.com/valsdav/zuko@master'
RUN python -m pip install 'mdmm @ git+https://github.com/the-moliver/mdmm@master'
