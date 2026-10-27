from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
LABELS = (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)

Label = Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


class Document(BaseModel):
    doc_id: str
    title: str
    sentences: list[str]


class Claim(BaseModel):
    claim_id: str
    text: str
    label: Optional[Label] = None
    gold_doc_ids: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str


class Hit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


class Verdict(BaseModel):
    label: Label
    reasoning: str = ""
    citations: list[str] = Field(default_factory=list)


class CheckResult(BaseModel):
    claim: str
    label: Label
    reasoning: str
    sources: list[Hit] = Field(default_factory=list)
    top_score: float = 0.0
    abstained: bool = False
    abstain_reason: Optional[str] = None
    queries: list[str] = Field(default_factory=list)
    attempts: int = 0
    timings: dict[str, float] = Field(default_factory=dict)


class AuditedClaim(BaseModel):
    index: int
    claim: str
    section: int = 0
    result: Optional[CheckResult] = None
    wording_flags: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    @property
    def label(self) -> Optional[str]:
        return self.result.label if self.result else None

    @property
    def possibly_overstated(self) -> bool:
        return bool(self.wording_flags) and self.label != SUPPORTED


class AuditReport(BaseModel):
    items: list[AuditedClaim] = Field(default_factory=list)
    n_sections: int = 0
    n_found: int = 0
    n_checked: int = 0
    sections_truncated: bool = False
    claims_truncated: bool = False

    def counts(self) -> dict[str, int]:
        tally = {SUPPORTED: 0, REFUTED: 0, NOT_ENOUGH_INFO: 0, "ERROR": 0}
        for item in self.items:
            tally[item.label or "ERROR"] += 1
        return tally

    def flagged(self) -> list[AuditedClaim]:
        return [i for i in self.items if i.label == REFUTED or i.possibly_overstated]


class AgentReply(BaseModel):
    text: str
    route: str
    result: Optional[CheckResult] = None
