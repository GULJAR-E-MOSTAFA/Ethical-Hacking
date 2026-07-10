import logging
import re
from typing import Optional
from dataclasses import dataclass
import unicodedata

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    lowercase: bool = False
    remove_punctuation: bool = False
    remove_numbers: bool = False
    remove_extra_whitespace: bool = True
    remove_accents: bool = False
    normalize_unicode: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_html_tags: bool = False

class TextPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)")
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.html_pattern = re.compile(r"<[^>]+>")

    def preprocess(self, text: str, config: Optional[PreprocessingConfig] = None) -> str:
        if config is None:
            config = PreprocessingConfig()

        result = text

        if config.normalize_unicode:
            result = unicodedata.normalize("NFKD", result)

        if config.remove_html_tags:
            result = self.html_pattern.sub("", result)

        if config.remove_urls:
            result = self.url_pattern.sub("", result)

        if config.remove_emails:
            result = self.email_pattern.sub("", result)

        if config.remove_accents:
            result = "".join(c for c in unicodedata.normalize("NFD", result) if unicodedata.category(c) != "Mn")

        if config.lowercase:
            result = result.lower()

        if config.remove_numbers:
            result = re.sub(r"\d+", "", result)

        if config.remove_punctuation:
            result = re.sub(r"[^\w\s]", "", result)

        if config.remove_extra_whitespace:
            result = re.sub(r"\s+", " ", result).strip()

        return result

    def clean_text(self, text: str) -> str:
        return self.preprocess(text)
