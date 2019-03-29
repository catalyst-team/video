FROM anibali/pytorch:cuda-9.0

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PYTHONPATH tsn
ENV FLASK_APP server/app.py

ADD requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt
RUN pip install git+git://github.com/catalyst-team/catalyst.git@379c27bb2c8a34ca4a53794166f44d427c96bd29

USER root
RUN apt-get update && apt-get install -y libsm6 libxext6 libturbojpeg ffmpeg

ADD . /app