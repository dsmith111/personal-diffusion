FROM python:3.8-slim

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc curl      ca-certificates python3 && \
apt clean && rm -rf /var/lib/apt/lists/*

RUN apt -y update && apt install -y wget nano git build-essential yasm pkg-config

# Compile and install ffmpeg from source
RUN git clone https://github.com/FFmpeg/FFmpeg /root/ffmpeg && \
    cd /root/ffmpeg && \
    ./configure --enable-nonfree --disable-shared --extra-cflags=-I/usr/local/include && \
    make -j8 && make install -j8

RUN apt install ffmpeg libsm6 libxext6  -y
RUN pip install -U setuptools pip

COPY . /queue
WORKDIR /queue
RUN pip install -r requirements.txt

RUN git clone https://github.com/ModeratePrawn/stable-diffusion-cpu.git /queue/stable-diffusion-repo
RUN rm ./stable-diffusion-repo/scripts/txt2img.py && cp ./sd-mod-files/txt2img.py ./stable-diffusion-repo/scripts/txt2img.py
RUN echo "" >> ./stable-diffusion-repo/scripts/__init__.py
RUN mkdir ./stable-diffusion-repo/models/ldm/stable-diffusion-v1 && \ 
	wget https://files.stablediffusionhub.com/model/sd-v1-4.ckpt -O ./stable-diffusion-repo/models/ldm/stable-diffusion-v1/model.ckpt
