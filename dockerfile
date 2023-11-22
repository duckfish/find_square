FROM python:3.11-slim-buster

WORKDIR /find_square

COPY ./app /find_square/app
COPY ./static /find_square/static
COPY ./templates /find_square/templates
COPY ./logs /find_square/logs
COPY ./requirements.txt /find_square/requirements.txt
COPY ./SquareNet_2111v2_updated.h5 /find_square/SquareNet_2111v2_updated.h5

RUN apt-get update && \
    apt-get install --no-install-recommends -y libffi-dev ffmpeg libsm6 libxext6 && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python3", "/find_square/app/main.py"]