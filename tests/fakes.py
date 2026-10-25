from __future__ import annotations

import hashlib
import re

import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from factcheck.schemas import Document, Hit


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HashEmbedder:
    dim = 128

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype="float32")
        for token in tokens(text):
            index = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts):
        return np.stack([self._vector(t) for t in texts]).astype("float32")

    def embed_query(self, text):
        return self._vector(text)


class OverlapReranker:
    def rerank(self, query, hits, top_n):
        query_tokens = set(tokens(query))
        scored = []
        for hit in hits:
            overlap = len(query_tokens & set(tokens(hit.text))) / max(len(query_tokens), 1)
            scored.append((overlap, hit))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [h.model_copy(update={"score": float(s)}) for s, h in scored[:top_n]]


class FakeLLM:
    def __init__(
        self,
        verdict_reply: str,
        followup_reply: str = "Because the first passage states it directly.",
        think: bool = False,
    ):
        self.verdict_reply = verdict_reply
        self.followup_reply = followup_reply
        self.think = think
        self.calls: list[str] = []
        self.runnable = RunnableLambda(self._respond)

    def kinds(self) -> list[str]:
        return self.calls

    def _message(self, content: str) -> AIMessage:
        prefix = '<think>pondering {"verdict": "REFUTED"} first</think>\n' if self.think else ""
        return AIMessage(content=prefix + content)

    def _respond(self, prompt_value):
        text = prompt_value.to_string()
        if "You classify the latest message" in text:
            self.calls.append("router")
            message = text.rsplit("Latest message:", 1)[-1].lower()
            if "hello" in message:
                return self._message("OTHER")
            if "why" in message:
                return self._message("FOLLOWUP")
            return self._message("CLAIM")
        if "You write search queries" in text:
            self.calls.append("rewrite")
            return self._message('{"queries": ["alternative phrasing one", "alternative phrasing two"]}')
        if "You are a careful scientific fact-checker" in text:
            self.calls.append("judge")
            return self._message(self.verdict_reply)
        if "You answer follow-up questions" in text:
            self.calls.append("followup")
            return self._message(self.followup_reply)
        self.calls.append("unknown")
        return self._message("")


def sample_docs() -> list[Document]:
    return [
        Document(
            doc_id="1",
            title="Vitamin D and respiratory infection",
            sentences=[
                "Vitamin D supplementation reduced the risk of acute respiratory infection in a meta-analysis.",
                "The effect was strongest in people with low baseline vitamin D levels.",
            ],
        ),
        Document(
            doc_id="2",
            title="Statins and cardiovascular outcomes",
            sentences=[
                "Statin therapy lowers LDL cholesterol and reduces major cardiovascular events.",
                "Adverse muscle symptoms were uncommon in randomized trials.",
            ],
        ),
        Document(
            doc_id="3",
            title="Exercise and bone density",
            sentences=[
                "Resistance training increased bone mineral density in postmenopausal women.",
                "Aerobic exercise alone showed a smaller effect.",
            ],
        ),
    ]
