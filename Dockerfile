FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY neurotk /app/neurotk

RUN python -m pip install --upgrade pip \
    && pip install .

EXPOSE 8000

CMD ["uvicorn", "neurotk.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
