import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]

class EntityValidator:
    def __init__(self):
        self.entity_type_patterns = {
            "PERSON": r"^[A-Z][a-z]+(\s[A-Z][a-z]+)*$",
            "ORG": r"^[A-Z][a-zA-Z0-9\s&.,'-]*$",
            "GPE": r"^[A-Z][a-zA-Z0-9\s'-]*$",
        }

    def validate_entity(self, entity: Dict[str, Any]) -> ValidationResult:
        issues = []
        warnings = []
        suggestions = []

        required_fields = ["text", "label", "start_char", "end_char"]
        for field in required_fields:
            if field not in entity:
                issues.append(f"Missing required field: {field}")

        if issues:
            return ValidationResult(is_valid=False, issues=issues, warnings=warnings, suggestions=suggestions)

        text = entity.get("text", "")
        label = entity.get("label", "")

        if not text or not text.strip():
            issues.append("Entity text is empty")

        if len(text) > 100:
            warnings.append(f"Entity text is very long ({len(text)} chars)")

        if "  " in text:
            suggestions.append("Entity contains multiple spaces")

        start = entity.get("start_char", 0)
        end = entity.get("end_char", 0)
        if start >= end:
            issues.append(f"Invalid positions: start={start}, end={end}")

        confidence = entity.get("confidence", 1.0)
        if confidence < 0 or confidence > 1:
            issues.append(f"Invalid confidence: {confidence}")
        elif confidence < 0.7:
            warnings.append(f"Low confidence: {confidence}")

        is_valid = len(issues) == 0
        return ValidationResult(is_valid=is_valid, issues=issues, warnings=warnings, suggestions=suggestions)

    def validate_batch(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = [self.validate_entity(e) for e in entities]
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count

        all_issues = []
        all_warnings = []
        all_suggestions = []

        for result in results:
            all_issues.extend(result.issues)
            all_warnings.extend(result.warnings)
            all_suggestions.extend(result.suggestions)

        return {
            "total_entities": len(entities),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "validity_rate": (valid_count / len(entities) * 100) if entities else 0,
            "issues": all_issues,
            "warnings": all_warnings,
            "suggestions": all_suggestions,
        }

    def check_for_overlaps(self, entities: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        overlaps = []
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i + 1 :]:
                start1, end1 = ent1.get("start_char", 0), ent1.get("end_char", 0)
                start2, end2 = ent2.get("start_char", 0), ent2.get("end_char", 0)
                if not (end1 <= start2 or end2 <= start1):
                    overlaps.append((ent1, ent2))
        return overlaps
