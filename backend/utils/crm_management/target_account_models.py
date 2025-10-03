from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class TargetAccount:
    id: str
    name: str
    industry: Optional[str] = None
    size: Optional[str] = None
    region: Optional[str] = None
    contacts: List[dict] = field(default_factory=list)
    status: Optional[str] = None
    notes: Optional[str] = ""