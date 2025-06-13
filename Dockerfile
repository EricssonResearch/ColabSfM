# starting image, preference given to runtime instead of base
FROM ubuntu:22.04 as compiler 
LABEL maintainer="Johan Edstedt johan.edstedt@liu.se"
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
RUN conda create -n colabsfm python=3.10
ENV HOME="/"
RUN echo "source activate colabsfm" > ~/.bashrc
ENV PATH="/opt/conda/envs/colabsfm/bin:${PATH}"
# RUN conda install -c conda-forge cxx-compiler
WORKDIR /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN conda install -c "nvidia/label/cuda-12.2.0" cuda


WORKDIR /app/
COPY pretrained /app/pretrained
# copy stuff to workdir, want to do this last for cache
COPY third_party /app/third_party
#COPY third_party/RoITr/cpp_wrappers/pointops/pointops-0.0.0-cp310-cp310-linux_x86_64.whl /app/
COPY .gitignore setup.py /app/
WORKDIR /app/third_party/pointops
RUN pip install pointops-0.0.0-cp310-cp310-linux_x86_64.whl

# Install pointops
#RUN pip install .
#RUN apt update && apt install -y build-essential
# Install colabsfm
WORKDIR /app/
COPY colabsfm /app/colabsfm
RUN pip install .
COPY experiments /app/experiments
