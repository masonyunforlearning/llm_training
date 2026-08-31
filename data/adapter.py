from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class Document:
    text: str
    source: str
    metadata: Optional[Dict[str, Any]] = None

def normalize_record(record, text_field="text", source="unknown"):
    text = record.get(text_field)
    if not isinstance(text, str) or not text.strip():
        return None
    return Document(text=text.strip(), source=source, metadata=record)
