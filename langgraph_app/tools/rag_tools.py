"""
LangGraph LifeScienceBench — RAG Tools.
Dependency-free hybrid retrieval over the local corpus.
Uses vendored bench functions (no external lifesciencebench dependency).
"""

from typing import Any

from ._bench_vendor import retrieve_hybrid, ingest, parse
from .registry import ToolMeta, register_tool
from .knowledge_tools import make_knowledge_tool, LITERATURE_PROMPT
from langgraph_app.config import MAX_RETRIEVAL_CHUNKS


# ── Tool 10: Hybrid Retrieval ──────────────────────────────────
def _hybrid_retrieval(question: str = "", k: int = MAX_RETRIEVAL_CHUNKS, **kwargs) -> dict[str, Any]:
    """Retrieve relevant document chunks using hybrid TF-IDF + character n-gram scoring."""
    result: dict[str, Any] = {
        "tool": "hybrid_retrieval",
        "status": "abstain",
        "summary": "",
        "data": {"chunks": []},
        "warnings": [],
    }

    if not question:
        result["summary"] = "No question provided for retrieval."
        return result

    try:
        chunks = retrieve_hybrid(question, k=k)

        result["data"]["chunks"] = [
            {
                "citation": e.citation,
                "document": e.document,
                "excerpt": e.excerpt[:300] + ("..." if len(e.excerpt) > 300 else ""),
                "score": e.score,
            }
            for e in chunks
        ]
        result["data"]["total_chunks"] = len(chunks)

        if chunks:
            result["status"] = "success"
            docs_found = len(set(c["document"] for c in result["data"]["chunks"]))
            result["summary"] = f"Retrieved {len(chunks)} chunks from {docs_found} document(s)"
        else:
            result["status"] = "none"
            result["summary"] = "No local evidence retrieved. Substantive claims withheld."
    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Retrieval error: {str(e)}"

    return result


register_tool(ToolMeta(
    name="hybrid_retrieval",
    description="Retrieve relevant scientific text chunks from the local corpus using hybrid TF-IDF + character n-gram scoring. Returns cited evidence.",
    domain="rag",
    requires_input=["question"],
    produces="evidence",
    func=_hybrid_retrieval,
))


# ── Tool 11: Corpus Ingest ─────────────────────────────────────
def _corpus_ingest(text: str = "", filename: str = "uploaded_document.txt", **kwargs) -> dict[str, Any]:
    """Ingest a text document into the local corpus for later retrieval."""
    result: dict[str, Any] = {
        "tool": "corpus_ingest",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": [],
    }

    if not text:
        result["summary"] = "No text provided for ingestion."
        return result

    try:
        stored_name = ingest(filename, text)
        manifest = parse(filename, text)

        result["data"] = {
            "stored_as": stored_name,
            "manifest_kind": manifest.get("kind", "unknown"),
            "sha256": manifest.get("sha256", ""),
        }
        result["status"] = "success"
        result["summary"] = f"Ingested document as '{stored_name}'"
    except Exception as e:
        result["status"] = "error"
        result["summary"] = str(e)

    return result


register_tool(ToolMeta(
    name="corpus_ingest",
    description="Ingest a text document into the local corpus for retrieval. Provide the document text and an optional filename.",
    domain="rag",
    requires_input=["text", "filename"],
    produces="evidence",
    func=_corpus_ingest,
))


# ── Tool: Literature Knowledge (LLM-powered, ALWAYS fires) ───
register_tool(make_knowledge_tool(
    domain="literature",
    tool_name="literature_knowledge",
    description="Answer ANY literature question: landmark papers, systematic review strategy, critical appraisal, evidence grading — open-ended. Always runs.",
    domain_label="Scientific Literature",
    system_prompt=LITERATURE_PROMPT,
))
