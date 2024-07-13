# starting image, preference given to runtime instead of base
#FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04 as compiler
FROM ubuntu:22.04 as compiler 
LABEL maintainer="Johan Edstedt johan.edstedt@ericsson.com"
ARG TIMEZONE="Sweden/Stockholm" 

# install dependencies (the last 3 are needed for opencv-python)
RUN apt update && apt install -y python3 python3-venv ffmpeg libsm6 libxext6 gcc g++
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:${PATH}"
ARG PATH="/opt/conda/bin:${PATH}"

WORKDIR /opt
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /opt/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# define a working directory inside the docker image, it will contain your code
RUN conda create -n sfmreg python=3.10
ENV HOME="/"
RUN echo "source activate sfmreg" > ~/.bashrc
ENV PATH="/opt/conda/envs/sfmreg/bin:${PATH}"
# RUN conda install -c conda-forge cxx-compiler
WORKDIR /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN conda install -c "nvidia/label/cuda-12.2.0" cuda


# # install the EGAD Certificates
# ENV ca_path /usr/share/ca-certificates/ericsson/
# RUN mkdir -p $ca_path
# COPY ca-certs/* $ca_path
# RUN echo "ericsson/EGADIssuingCA3.crt" >> /etc/ca-certificates.conf
# RUN echo "ericsson/EGADRootCA.crt" >> /etc/ca-certificates.conf
# RUN update-ca-certificates

WORKDIR /app/
COPY pretrained /app/pretrained
# copy stuff to workdir, want to do this last for cache
COPY third_party /app/third_party
#COPY third_party/RoITr/cpp_wrappers/pointops/pointops-0.0.0-cp310-cp310-linux_x86_64.whl /app/
COPY .gitignore setup.py /app/
WORKDIR /app/third_party/pointops
RUN pip install pointops-0.0.0-cp310-cp310-linux_x86_64.whl

# install geotransformer
WORKDIR /app/
COPY GeoTransformer /app/GeoTransformer
WORKDIR /app/GeoTransformer
# RUN pip install -r requirements.txt
RUN python setup.py build develop

# Install pointops
#RUN pip install .
#RUN apt update && apt install -y build-essential
# Install sfmreg
WORKDIR /app/
COPY sfmreg /app/sfmreg
RUN pip install .
COPY experiments /app/experiments

# install Predator
WORKDIR /app/sfmreg/OverlapPredator/cpp_wrappers
RUN sh compile_wrappers.sh
WORKDIR /app/