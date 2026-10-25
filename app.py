from __future__ import annotations

import uuid

import streamlit as st

from factcheck.config import get_settings
from factcheck.factory import build_agent
from factcheck.llm import active_model_name
from factcheck.observability import flush, langfuse_enabled

st.set_page_config(page_title="Claim Checker", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading models and index...")
def load_agent():
    return build_agent()


settings = get_settings()
agent = load_agent()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Claim Checker")
    st.caption("Checks scientific claims against research abstracts and cites its evidence.")
    st.write(f"LLM: {settings.llm_provider} / {active_model_name(settings)}")
    st.write(f"Vector store: {settings.vector_backend}")
    st.write(f"Tracing: {'on' if langfuse_enabled() else 'off'}")
    if st.button("New conversation"):
        agent.clear(st.session_state.session_id)
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()
    st.caption("Research tool only. Not medical advice.")

st.title("Claim Checker")


def render_result(result):
    if result is None:
        return
    with st.expander(f"Evidence and details ({len(result.sources)} passages)"):
        st.write(f"Top relevance score: {result.top_score:.3f} | Retrieval rounds: {result.attempts}")
        if result.queries:
            st.write("Queries used: " + " | ".join(result.queries))
        for hit in result.sources:
            st.markdown(f"**{hit.title or 'Untitled'}** (doc {hit.doc_id}, score {hit.score:.3f})")
            st.write(hit.text)
        if result.timings:
            st.caption("Latency: " + ", ".join(f"{k} {v}s" for k, v in result.timings.items()))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message["role"] == "assistant":
            render_result(message.get("result"))

prompt = st.chat_input("Enter a claim, or ask a follow-up question")
if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking..."):
            reply = agent.ask(st.session_state.session_id, prompt)
        st.markdown(reply.text)
        render_result(reply.result)
    st.session_state.messages.append({"role": "assistant", "text": reply.text, "result": reply.result})
    flush()
