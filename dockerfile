FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY ./src /app
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
COPY ./templates /app/templates
COPY ./model/random_forest_model.joblib /app/model/random_forest_model.joblib

RUN uv sync --locked
# Presuming there is a `my_app` command provided by the project
CMD ["uv", "run", "app.py"]