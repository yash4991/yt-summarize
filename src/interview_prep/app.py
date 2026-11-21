"""Streamlit web UI.  Run with:  interview-prep ui   (or: streamlit run src/interview_prep/app.py)"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import streamlit as st

from interview_prep import render
from interview_prep.config import Settings, missing_api_key
from interview_prep.pipeline import BatchReport, run_batch

st.set_page_config(page_title="Interview Prep Agent", page_icon="🎯", layout="wide")

base = Settings()

# --------------------------------------------------------------------------- #
# Sidebar: settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", base.model, help="Any CrewAI-supported model, e.g. anthropic/claude-sonnet-5-5")
    if key_var := missing_api_key(model):
        st.error(f"{key_var} is not set. Add it to your .env file (only cached videos will work).")
    st.caption("Always saved: .txt files (short, detailed, Q&A)")
    exp_md = st.checkbox("Markdown (.md)", "md" in base.exports)
    exp_pdf = st.checkbox("PDF study guide", "pdf" in base.exports)
    notion_ready = bool(base.notion_token and base.notion_parent_page_id)
    exp_notion = st.checkbox("Publish to Notion", "notion" in base.exports and notion_ready, disabled=not notion_ready,
                             help=None if notion_ready else "Set NOTION_TOKEN and NOTION_PARENT_PAGE_ID in .env")
    st.divider()
    force = st.checkbox("Ignore cache (regenerate)", False)
    whole_playlist = st.checkbox("Treat watch?v=…&list=… links as whole playlists", False)
    limit = st.number_input("Max videos per playlist (0 = all)", min_value=0, value=0, step=1)
    st.divider()
    st.caption(f"Output folder: `{base.output_dir.resolve()}`")

settings = base.with_overrides(
    model=model,
    exports=tuple(x for x, on in [("md", exp_md), ("pdf", exp_pdf), ("notion", exp_notion)] if on),
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def show_pack(pack, meta, folder: Path, key: str) -> None:
    tabs = st.tabs(["Short summary", "Detailed summary", "Interview Q&A", "Quality review", "Files"])
    with tabs[0]:
        st.markdown(render.render_short(pack, meta, "md"))
    with tabs[1]:
        st.markdown(render.render_detailed(pack, meta, "md"))
    with tabs[2]:
        st.markdown(render.render_qa(pack, meta, "md"))
    with tabs[3]:
        r = pack.review
        c = st.columns(4)
        c[0].metric("Overall", f"{r.overall}/10")
        c[1].metric("Faithfulness", f"{r.faithfulness_score}/10")
        c[2].metric("Coverage", f"{r.coverage_score}/10")
        c[3].metric("Clarity", f"{r.clarity_score}/10")
        st.write(r.verdict)
        if pack.revisions:
            st.info(f"Revised {pack.revisions}x after the reviewer's first pass.")
        for i in r.issues:
            st.markdown(f"- **{i.severity}** · {i.location}: {i.problem} → _{i.fix}_")
    with tabs[4]:
        st.code(str(folder), language=None)
        for f in sorted(folder.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                mime = "application/pdf" if f.suffix == ".pdf" else "text/plain"
                st.download_button(f"⬇ {f.name}", f.read_bytes(), file_name=f.name, mime=mime, key=f"{key}-{f.name}")


def load_library(root: Path) -> list[dict]:
    items = []
    for meta_file in root.rglob("metadata.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            data["_folder"] = meta_file.parent
            items.append(data)
        except (OSError, ValueError):
            continue
    return sorted(items, key=lambda d: d.get("generated_at", ""), reverse=True)


def run_in_background(urls: list[str], group: str | None) -> BatchReport:
    """Run the batch in a worker thread and stream its progress messages into the page."""
    q: queue.Queue[str] = queue.Queue()
    box: dict = {}

    def work() -> None:
        try:
            box["report"] = run_batch(urls, settings, force=force, group=group or None,
                                      playlist_limit=int(limit) or None, expand_playlists=whole_playlist,
                                      progress=q.put)
        except Exception as e:  # surfaced below
            box["error"] = e

    t = threading.Thread(target=work, daemon=True)
    with st.status("Working...", expanded=True) as status:
        t.start()
        while t.is_alive() or not q.empty():
            try:
                status.write(q.get(timeout=0.3))
            except queue.Empty:
                pass
        t.join()
        if "error" in box:
            status.update(label="Failed", state="error")
            raise box["error"]
        rep: BatchReport = box["report"]
        state = "error" if rep.failures and not rep.results else "complete"
        status.update(label=f"Done: {len(rep.results)} succeeded, {len(rep.failures)} failed", state=state,
                      expanded=False)
    return rep


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("🎯 Interview Prep Agent")
st.write("Paste YouTube videos or playlists. An AI interview coach turns each one into a short summary, "
         "a detailed study guide and likely interview questions, checked by an AI reviewer.")

new_tab, library_tab = st.tabs(["New", "Library"])

with new_tab:
    urls_text = st.text_area("YouTube URLs (one per line)", height=120,
                             placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/playlist?list=...")
    group = st.text_input("Group name (optional)",
                          help="Put these videos in one folder with a combined study guide. "
                               "Playlists are grouped automatically.")
    if st.button("Generate study notes", type="primary", disabled=not urls_text.strip()):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        st.session_state["report"] = run_in_background(urls, group)

    report: BatchReport | None = st.session_state.get("report")
    if report:
        for f in report.failures:
            st.error(f"**{f.url}**: {f.error}")
        if report.results:
            fresh = [r for r in report.results if not r.stats.from_cache]
            c = st.columns(4)
            c[0].metric("Videos", len(report.results))
            c[1].metric("From cache", len(report.results) - len(fresh))
            c[2].metric("Tokens used", f"{report.total_tokens:,}")
            avg = sum(r.pack.review.overall for r in report.results) / len(report.results)
            c[3].metric("Avg quality", f"{avg:.1f}/10")
        for g in report.guides:
            st.success(f"Combined study guide: `{g}`")
        for i, r in enumerate(report.results):
            with st.expander(f"{r.meta.title} · quality {r.pack.review.overall}/10", expanded=len(report.results) == 1):
                if r.notion_url:
                    st.markdown(f"[Open in Notion]({r.notion_url})")
                show_pack(r.pack, r.meta, Path(r.folder), key=f"new-{i}")

with library_tab:
    items = load_library(settings.output_dir)
    if not items:
        st.info("Nothing generated yet.")
    query = st.text_input("Filter", placeholder="Search titles...")
    for i, d in enumerate(items):
        title = d.get("video", {}).get("title", d["_folder"].name)
        if query and query.lower() not in title.lower():
            continue
        q = d.get("quality", {})
        with st.expander(f"{title} · quality {q.get('overall', '?')}/10 · {d.get('generated_at', '')[:10]}"):
            folder: Path = d["_folder"]
            docs = sorted(folder.glob("*.md")) or sorted(folder.glob("*.txt"))
            if docs:
                names = [p.name for p in docs]
                pick = st.selectbox("Document", names, key=f"lib-sel-{i}")
                text = (folder / pick).read_text(encoding="utf-8")
                st.markdown(text) if pick.endswith(".md") else st.text(text)
            st.code(str(folder), language=None)

st.caption(f"Model: {settings.model} · Rendered {time.strftime('%Y-%m-%d %H:%M')}")
