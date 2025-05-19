from typing import Dict,Optional,List
from dataclasses import dataclass, field

@dataclass
class PageQAData:
    """
    Stores data related to Q&A extraction for a single PDF page.
    """
    doc_id: str  # Document ID or path
    page_number: int  # 1-indexed PDF page number
    page_text: str
    qa_pairs: Optional[List[Dict[str, str]]] = field(default_factory=list) # Parsed JSON from LLM
    raw_response: Optional[str] = None # Raw string from LLM