import re
from typing import List, Dict, Any, Pattern, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PatternRule:
    name: str
    label: str
    pattern: Pattern
    confidence: float = 0.9
    metadata: Dict[str, Any] = None

class PatternMatcher:
    def __init__(self):
        self.rules: Dict[str, PatternRule] = {}
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        self.add_rule("email", "EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", confidence=0.95)
        self.add_rule("url", "URL", r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)", confidence=0.95)
        self.add_rule("ip_address", "IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", confidence=0.9)
        self.add_rule("phone_us", "PHONE", r"(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b", confidence=0.85)
        self.add_rule("date", "DATE", r"\b(?:(?:\d{4}-\d{2}-\d{2})|(?:\d{1,2}/\d{1,2}/\d{4}))\b", confidence=0.9)
        self.add_rule("hashtag", "HASHTAG", r"#\w+", confidence=0.95)
        self.add_rule("mention", "MENTION", r"@\w+", confidence=0.95)

    def add_rule(self, name: str, label: str, pattern: str, confidence: float = 0.9, metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            compiled_pattern = re.compile(pattern)
            rule = PatternRule(name=name, label=label, pattern=compiled_pattern, confidence=confidence, metadata=metadata or {})
            self.rules[name] = rule
            logger.info(f"Pattern rule added: {name}")
        except re.error as e:
            logger.error(f"Invalid regex pattern for rule {name}: {e}")

    def remove_rule(self, name: str) -> None:
        if name in self.rules:
            del self.rules[name]

    def match(self, text: str, rule_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        matches = []
        rules_to_use = ({n: self.rules[n] for n in rule_names if n in self.rules} if rule_names else self.rules)

        for rule_name, rule in rules_to_use.items():
            for match in rule.pattern.finditer(text):
                matches.append({
                    "text": match.group(),
                    "label": rule.label,
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "confidence": rule.confidence,
                    "rule": rule_name,
                    "metadata": rule.metadata,
                })

        matches.sort(key=lambda x: x["start_char"])
        return matches

    def get_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "label": rule.label,
                "confidence": rule.confidence,
                "pattern": rule.pattern.pattern,
                "metadata": rule.metadata,
            }
            for name, rule in self.rules.items()
        }
