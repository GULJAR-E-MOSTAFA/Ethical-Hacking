from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import logging
import json
from datetime import datetime
import time

logger = logging.getLogger(__name__)

try:
    from ..core.ner_engine import NEREngine, Entity, ProcessingResult
    from ..core.pattern_matcher import PatternMatcher
    from ..core.entity_analyzer import EntityAnalyzer
    from ..core.entity_validator import EntityValidator
    from ..core.text_preprocessor import TextPreprocessor, PreprocessingConfig
except ImportError:
    pass

class EntitySchema(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = {}

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)
    model: str = "en_core_web_sm"
    include_patterns: bool = False
    return_tokens: bool = False

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    model: str = "en_core_web_sm"
    batch_size: int = 50

class PredictResponse(BaseModel):
    text: str
    entities: List[EntitySchema]
    processing_time: float
    model_used: str
    language: str = "en"
    tokens: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    cache_stats: Optional[Dict[str, Any]] = None

class StatsResponse(BaseModel):
    total_processed: int
    total_entities_found: int
    cache_enabled: bool
    cache_stats: Optional[Dict[str, Any]] = None

def create_app() -> FastAPI:
    app = FastAPI(
        title="Advanced NER API",
        description="Named Entity Recognition API",
        version="1.0.0",
    )

    ner_engine = None
    pattern_matcher = PatternMatcher()
    entity_analyzer = EntityAnalyzer()
    entity_validator = EntityValidator()
    text_preprocessor = TextPreprocessor()

    @app.on_event("startup")
    async def startup():
        nonlocal ner_engine
        try:
            logger.info("Initializing NER Engine...")
            ner_engine = NEREngine(model_name="en_core_web_sm", enable_cache=True, cache_size=1000)
            logger.info("NER Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        if ner_engine is None:
            raise HTTPException(status_code=503, detail="NER Engine not initialized")
        return HealthResponse(
            status="ok",
            timestamp=datetime.now().isoformat(),
            model_loaded=ner_engine.nlp is not None,
            cache_stats=ner_engine.get_cache_stats(),
        )

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats():
        if ner_engine is None:
            raise HTTPException(status_code=503, detail="NER Engine not initialized")
        stats = ner_engine.get_stats()
        return StatsResponse(
            total_processed=stats.get("total_processed", 0),
            total_entities_found=stats.get("total_entities_found", 0),
            cache_enabled=stats.get("cache_enabled", False),
            cache_stats=stats.get("cache_stats"),
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        if ner_engine is None:
            raise HTTPException(status_code=503, detail="NER Engine not initialized")
        try:
            start_time = time.time()
            result = ner_engine.process(request.text)
            processing_time = time.time() - start_time
            entities = [EntitySchema(**e.to_dict()) for e in result.entities]
            return PredictResponse(
                text=request.text,
                entities=entities,
                processing_time=processing_time,
                model_used=result.model_used,
                language=result.language,
                tokens=result.tokens if request.return_tokens else None,
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch")
    async def batch_predict(request: BatchPredictRequest):
        if ner_engine is None:
            raise HTTPException(status_code=503, detail="NER Engine not initialized")
        try:
            results = ner_engine.process_batch(request.texts, batch_size=request.batch_size)
            response_data = []
            for result in results:
                response_data.append({
                    "text": result.text,
                    "entities": [e.to_dict() for e in result.entities],
                    "processing_time": result.processing_time,
                })
            return {"results": response_data}
        except Exception as e:
            logger.error(f"Batch error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/file")
    async def predict_from_file(file: UploadFile = File(...)):
        if ner_engine is None:
            raise HTTPException(status_code=503, detail="NER Engine not initialized")
        try:
            content = await file.read()
            text = content.decode("utf-8")
            result = ner_engine.process(text)
            return PredictResponse(
                text=result.text,
                entities=[EntitySchema(**e.to_dict()) for e in result.entities],
                processing_time=result.processing_time,
                model_used=result.model_used,
            )
        except Exception as e:
            logger.error(f"File error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze")
    async def analyze_entities(request: PredictRequest):
        if ner_engine is None:
            raise HTTPException(status_code=503, detail="NER Engine not initialized")
        try:
            result = ner_engine.process(request.text)
            entities_dicts = [e.to_dict() for e in result.entities]
            stats = entity_analyzer.analyze(entities_dicts)
            return {
                "total_entities": stats.total_entities,
                "unique_entities": stats.unique_entities,
                "entity_types": stats.entity_types,
                "entity_type_distribution": stats.entity_type_distribution,
                "most_common_entities": stats.most_common_entities,
                "entity_lengths": stats.entity_lengths,
            }
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/models")
    async def list_models():
        if ner_engine is None:
            raise HTTPException(status_code=503, detail="NER Engine not initialized")
        return {"models": list(ner_engine.SUPPORTED_MODELS.keys())}

    @app.post("/cache/clear")
    async def clear_cache():
        if ner_engine is None:
            raise HTTPException(status_code=503, detail="NER Engine not initialized")
        ner_engine.clear_cache()
        return {"message": "Cache cleared"}

    return app

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
