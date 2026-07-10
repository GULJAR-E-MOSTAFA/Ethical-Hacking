# NER Tool (CLI + FastAPI)

This package provides:
- A CLI for quick NER runs (ner_cli.py)
- A FastAPI web server with /predict endpoint (app/main.py)
- Dockerfile to run the API in a container

Setup (local)
1. Create and activate a virtualenv:
   python3 -m venv .venv
   source .venv/bin/activate

2. Install:
   pip install -r requirements.txt

3. Download spaCy model:
   python -m spacy download en_core_web_sm

CLI usage
- JSON output:
  python ner_cli.py --text "Apple is buying a U.K. startup" --format json

- Plain list:
  python ner_cli.py --file sample.txt --format plain

API usage
- Start server:
  uvicorn app.main:app --reload

- Health check:
  curl http://localhost:8000/health

- Predict by text:
  curl -X POST -F 'text=Barack Obama was born in Hawaii.' http://localhost:8000/predict

- Predict by file:
  curl -X POST -F 'file=@article.txt' http://localhost:8000/predict

Docker
- Build:
  docker build -t ner-service:latest .

- Run:
  docker run -p 8000:8000 ner-service:latest

Notes / next steps
- If you want the transformer-based model (better accuracy), change model name to `en_core_web_trf` and adjust Dockerfile (requires CUDA or more resources).
- I can add unit tests, CI workflow, or a small web UI if you want.
