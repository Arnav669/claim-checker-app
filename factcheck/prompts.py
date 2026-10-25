from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify the latest message in a fact-checking chat. "
            "Answer with exactly one word.\n"
            "CLAIM: a factual statement or question that should be verified.\n"
            "FOLLOWUP: asks about the previous verdict, its reasoning, its sources, "
            "or asks for clarification about it.\n"
            "OTHER: greetings, thanks, or anything unrelated to checking a claim.",
        ),
        (
            "human",
            "Previous claim: {last_claim}\n"
            "Previous verdict: {last_verdict}\n"
            "Conversation so far:\n{history}\n\n"
            "Latest message: {message}",
        ),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write search queries for a search engine over scientific research abstracts. "
            "Given a claim and queries that already failed to find relevant evidence, write 2 new "
            "queries that use different key terms, synonyms, or the underlying scientific concepts. "
            'Reply with JSON only: {{"queries": ["...", "..."]}}',
        ),
        ("human", "Claim: {claim}\nFailed queries: {failed}"),
    ]
)

VERDICT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful scientific fact-checker. You are given a claim and numbered "
            "evidence passages retrieved from a corpus of research abstracts.\n"
            "Rules:\n"
            "- Use only the passages. Never use outside knowledge.\n"
            "- SUPPORTED: the passages directly state or clearly imply the claim is true.\n"
            "- REFUTED: the passages directly state or clearly imply the claim is false.\n"
            "- NOT_ENOUGH_INFO: the passages are related but do not settle the claim, or are unrelated.\n"
            '- Cite the ids of the passages your verdict relies on, for example ["P1", "P3"]. '
            "For NOT_ENOUGH_INFO the citation list may be empty.\n"
            "Reply with one JSON object and nothing else:\n"
            '{{"verdict": "SUPPORTED" or "REFUTED" or "NOT_ENOUGH_INFO", '
            '"reasoning": "two sentences at most", "citations": ["P1"]}}',
        ),
        ("human", "Claim: {claim}\n\nPassages:\n{passages}"),
    ]
)

FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer follow-up questions about a fact-check you just performed. "
            "Use only the claim, verdict, reasoning and passages below. If the answer is not in "
            "them, say so and suggest checking a new claim. Be concise.\n\n"
            "Claim: {claim}\n"
            "Verdict: {verdict}\n"
            "Reasoning: {reasoning}\n"
            "Passages:\n{passages}",
        ),
        ("human", "Conversation so far:\n{history}\n\nQuestion: {message}"),
    ]
)

OTHER_REPLY = (
    "I check scientific claims against a corpus of research abstracts. "
    "Send me a claim, for example: \"Vitamin D supplementation reduces the risk of respiratory infections.\" "
    "After a verdict you can ask me why, or ask to see the sources."
)
