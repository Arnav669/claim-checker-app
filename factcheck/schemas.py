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


class AgentReply(BaseModel):
    text: str
    route: str
    result: Optional[CheckResult] = None
