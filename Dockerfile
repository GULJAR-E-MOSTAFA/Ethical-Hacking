# Simple Dockerfile for the NER FastAPI app
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (en_core_web_sm). This increases image size; change if you prefer to download at runtime.
RUN python -m spacy download en_core_web_sm

COPY app /app/app
COPY ner_cli.py /app/ner_cli.py
COPY README-NER.md /app/README-NER.md

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
