import logging
from typing import List, Dict, Any, Tuple
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class EntityStatistics:
    total_entities: int
    unique_entities: int
    entity_types: Dict[str, int]
    entity_type_distribution: Dict[str, float]
    most_common_entities: List[Tuple[str, int]]
    entity_lengths: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class EntityAnalyzer:
    def __init__(self):
        self.entities_history: List[List[Dict[str, Any]]] = []

    def analyze(self, entities: List[Dict[str, Any]]) -> EntityStatistics:
        if not entities:
            return EntityStatistics(
                total_entities=0,
                unique_entities=0,
                entity_types={},
                entity_type_distribution={},
                most_common_entities=[],
                entity_lengths={},
            )

        total_entities = len(entities)
        unique_texts = set(e["text"] for e in entities)
        unique_entities = len(unique_texts)

        entity_types = Counter(e["label"] for e in entities)
        entity_types_dict = dict(entity_types)
        entity_type_distribution = {k: (v / total_entities * 100) for k, v in entity_types_dict.items()}

        entity_counter = Counter(e["text"] for e in entities)
        most_common = entity_counter.most_common(10)

        lengths = [len(e["text"]) for e in entities]
        entity_lengths = {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "avg": sum(lengths) / len(lengths) if lengths else 0,
        }

        stats = EntityStatistics(
            total_entities=total_entities,
            unique_entities=unique_entities,
            entity_types=entity_types_dict,
            entity_type_distribution=entity_type_distribution,
            most_common_entities=most_common,
            entity_lengths=entity_lengths,
        )

        self.entities_history.append([e for e in entities])
        return stats

    def get_entity_type_summary(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        summary = {}
        for entity in entities:
            label = entity["label"]
            if label not in summary:
                summary[label] = []
            summary[label].append(entity["text"])

        for label in summary:
            summary[label] = list(set(summary[label]))

        return summary
