FROM python:3.11-slim

RUN apt-get update && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

RUN rm -rf /tmp/*

WORKDIR /find_square

COPY ./app /find_square/app
COPY ./static /find_square/static
COPY ./templates /find_square/templates
COPY ./logs /find_square/logs
COPY ./requirements.txt /find_square/requirements.txt
COPY ./SquareNet_2111v2_updated.h5 /find_square/SquareNet_2111v2_updated.h5

ENTRYPOINT ["python3", "/find_square/app/main.py"]