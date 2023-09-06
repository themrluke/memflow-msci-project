FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
USER root

RUN apt-get update && apt-get install -y \
    git\
    vim\
    xrootd\
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN mkdir /opt/MEMFlow
ADD memflow /opt/MEMFlow/memflow
ADD .git /opt/MEMFlow/.git
ADD scripts /opt/MEMFlow/scripts
ADD setup.cfg /opt/MEMFlow/
ADD pyproject.toml /opt/MEMFlow/

WORKDIR /opt/MEMFlow
ARG PSEUDO_VERSION=1
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${PSEUDO_VERSION} python -m pip install -e .

RUN python -m pip install 'zuko @ git+https://github.com/valsdav/zuko@master'
RUN python -m pip install 'mdmm @ git+https://github.com/the-moliver/mdmm@master'
