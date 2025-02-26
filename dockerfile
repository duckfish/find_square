FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.6.3 /uv /uvx /bin/

RUN apt-get update && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /find_square

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --no-dev

COPY ./app /find_square/app
COPY ./static /find_square/static
COPY ./templates /find_square/templates
COPY ./requirements.txt /find_square/requirements.txt
COPY ./SquareNet_2111v2_updated.h5 /find_square/SquareNet_2111v2_updated.h5
COPY ./pyproject.toml /find_square/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

ENTRYPOINT ["python3", "/find_square/app/main.py"]