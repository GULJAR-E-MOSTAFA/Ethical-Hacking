import spacy
import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import hashlib
from functools import lru_cache
import time
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

@dataclass
class ProcessingResult:
    text: str
    entities: List[Entity]
    processing_time: float
    model_used: str
    language: str = "en"
    tokens: List[str] = field(default_factory=list)
    pos_tags: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "language": self.language,
            "tokens": self.tokens,
            "pos_tags": self.pos_tags,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

class NERCache:
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, ProcessingResult] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[ProcessingResult]:
        key = self._hash_text(text)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, text: str, result: ProcessingResult) -> None:
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        key = self._hash_text(text)
        self.cache[key] = result

    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
        }

class NEREngine:
    SUPPORTED_MODELS = {
        "en_core_web_sm": {"size": "small", "lang": "en", "vectors": False},
        "en_core_web_md": {"size": "medium", "lang": "en", "vectors": True},
        "en_core_web_lg": {"size": "large", "lang": "en", "vectors": True},
        "en_core_web_trf": {"size": "transformer", "lang": "en", "vectors": True},
    }

    def __init__(self, model_name: str = "en_core_web_sm", enable_cache: bool = True, cache_size: int = 1000, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.nlp = None
        self.cache = NERCache(max_size=cache_size) if enable_cache else None
        self.stats = {"total_processed": 0, "total_entities_found": 0, "cache_enabled": enable_cache}
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Model {self.model_name} loaded successfully")
        except OSError as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")

    def process(self, text: str, use_cache: bool = True) -> ProcessingResult:
        if use_cache and self.cache:
            cached_result = self.cache.get(text)
            if cached_result:
                return cached_result

        start_time = time.time()
        doc = self.nlp(text)

        entities = [
            Entity(text=ent.text, label=ent.label_, start_char=ent.start_char, end_char=ent.end_char, confidence=1.0)
            for ent in doc.ents
        ]

        tokens = [token.text for token in doc]
        pos_tags = [(token.text, token.pos_) for token in doc]
        processing_time = time.time() - start_time

        result = ProcessingResult(text=text, entities=entities, processing_time=processing_time, model_used=self.model_name, tokens=tokens, pos_tags=pos_tags)

        if self.cache:
            self.cache.set(text, result)

        self.stats["total_processed"] += 1
        self.stats["total_entities_found"] += len(entities)
        return result

    def process_batch(self, texts: List[str], batch_size: int = 50) -> List[ProcessingResult]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                result = self.process(text)
                results.append(result)
        return results

    def get_entity_types(self, text: str) -> Set[str]:
        result = self.process(text)
        return set(ent.label for ent in result.entities)

    def filter_entities(self, text: str, entity_types: List[str]) -> List[Entity]:
        result = self.process(text)
        return [ent for ent in result.entities if ent.label in entity_types]

    def get_cache_stats(self) -> Dict[str, Any]:
        if self.cache:
            return self.cache.stats()
        return {}

    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        if self.cache:
            stats["cache_stats"] = self.cache.stats()
        return stats

    def clear_cache(self) -> None:
        if self.cache:
            self.cache.clear()
