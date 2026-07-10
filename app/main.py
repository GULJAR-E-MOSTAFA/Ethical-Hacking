"""
FastAPI NER service.

Endpoints:
- GET /health
- POST /predict  (form-data: text OR file)
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import spacy
from typing import List, Optional

app = FastAPI(title="NER Service")

# load model once on startup
nlp = None

@app.on_event("startup")
def load_model():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            # Let the process start even if model missing; endpoints will return errors with guidance.
            nlp = None
            app.state.model_load_error = str(e)

class Entity(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int

class PredictResponse(BaseModel):
    text: str
    entities: List[Entity]

@app.get("/health")
def health():
    if getattr(app.state, "model_load_error", None):
        return {"status": "degraded", "model_error": app.state.model_load_error}
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    if not text and not file:
        raise HTTPException(status_code=400, detail="Provide 'text' form field or upload a 'file'.")
    if file:
        raw = await file.read()
        try:
            text = raw.decode('utf-8')
        except Exception:
            text = raw.decode('latin-1', errors='replace')

    if nlp is None:
        raise HTTPException(status_code=500, detail="spaCy model not loaded. Install and download 'en_core_web_sm' in server.")

    doc = nlp(text)
    ents = [
        {"text": ent.text, "label": ent.label_, "start_char": ent.start_char, "end_char": ent.end_char}
        for ent in doc.ents
    ]
    return {"text": text, "entities": ents}
