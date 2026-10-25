from __future__ import annotations

import os
from typing import Iterable, Optional


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def build_run_config(session_id: str, tags: Optional[Iterable[str]] = None) -> dict:
    config: dict = {
        "run_name": "factcheck",
        "metadata": {"langfuse_session_id": session_id},
    }
    if tags:
        config["tags"] = list(tags)
    if langfuse_enabled():
        try:
            from langfuse.langchain import CallbackHandler

            config["callbacks"] = [CallbackHandler()]
        except Exception:
            pass
    return config


def flush() -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass
