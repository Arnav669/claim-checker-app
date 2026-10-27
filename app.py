from __future__ import annotations

import uuid

import streamlit as st

from factcheck.config import get_settings
from factcheck.documents import extract_text
from factcheck.factory import build_system
from factcheck.formatting import format_audit_report, label_text
from factcheck.llm import active_model_name
from factcheck.observability import flush, langfuse_enabled

st.set_page_config(page_title="Claim Checker", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading models and index...")
def load_system():
    return build_system()


settings = get_settings()
agent, auditor = load_system()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Claim Checker")
    mode = st.radio("Mode", ["Chat", "Audit a document"])
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

BADGE = {"SUPPORTED": "✅", "REFUTED": "❌", "NOT_ENOUGH_INFO": "❔"}


def render_audit():
    st.subheader("Audit a document")
    st.caption(
        "Paste text or upload a PDF. The auditor pulls out checkable scientific claims "
        "and checks each one against the research abstracts."
    )
    uploaded = st.file_uploader("PDF or text file", type=["pdf", "txt", "md"])
    pasted = st.text_area("Or paste text", height=180)
    max_claims = st.slider("Maximum claims to check", 3, 15, settings.audit_max_claims)
    st.caption(
        f"The free model tier is rate limited, so each claim takes about "
        f"{max(settings.audit_pause_seconds, 5):.0f} seconds."
    )
    if st.button("Run audit", type="primary"):
        try:
            text = extract_text(uploaded.name, uploaded.getvalue()) if uploaded else pasted.strip()
            if not text:
                st.warning("Add some text or upload a file first.")
                text = None
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            text = None
        if text:
            bar = st.progress(0.0, text="Starting...")

            def on_progress(done, total, message):
                bar.progress(min(done / total, 1.0) if total else 1.0, text=message)

            st.session_state.audit_report = auditor.audit(text, max_claims, progress=on_progress)
            bar.empty()
            flush()

    report = st.session_state.get("audit_report")
    if report is None:
        return
    counts = report.counts()
    cols = st.columns(4)
    cols[0].metric("Checked", report.n_checked)
    cols[1].metric("Supported", counts["SUPPORTED"])
    cols[2].metric("Refuted", counts["REFUTED"])
    cols[3].metric("Not enough info", counts["NOT_ENOUGH_INFO"])
    if report.claims_truncated:
        st.info(f"{report.n_found} claims were found. A sample spread across the document was checked.")
    if report.sections_truncated:
        st.info("The document is long, so only the first part was read.")
    for item in report.items:
        icon = BADGE.get(item.label, "⚠️")
        flag = " ⚠ strong wording" if item.possibly_overstated else ""
        with st.expander(f"{icon} {item.index}. {item.claim}{flag}"):
            if item.error:
                st.error(f"Could not check this claim: {item.error}")
                continue
            st.markdown(f"**{label_text(item.label)}**")
            st.write(item.result.reasoning)
            if item.wording_flags:
                st.write("Strong wording: " + ", ".join(item.wording_flags))
            for hit in item.result.sources:
                st.markdown(f"- {hit.title or 'Untitled'} (doc {hit.doc_id}, score {hit.score:.3f})")
    st.download_button(
        "Download report (Markdown)",
        format_audit_report(report),
        file_name="audit_report.md",
        mime="text/markdown",
    )


if mode == "Audit a document":
    render_audit()
    st.stop()


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
