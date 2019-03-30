FROM anibali/pytorch:cuda-9.0

ADD requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

USER root
RUN apt-get update && apt-get install -y libsm6 libxext6 libturbojpeg ffmpeg

ADD . /app