ARG FROM_IMAGE=cmsml/cmsml:3.11
FROM ${FROM_IMAGE}

ADD requirements.txt /tmp/
RUN python -m pip install -r /tmp/requirements.txt

RUN python -m pip install 'zuko @ git+https://github.com/probabilists/zuko@master'
RUN python -m pip install 'mdmm @ git+https://github.com/the-moliver/mdmm@master'
