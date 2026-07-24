"""
LangGraph LifeScienceBench — Literature & Evidence Team (6 agents).

📄 paper_summarizer     — Summarize one or many papers
🔗 evidence_synthesizer  — Build consensus from literature
⚡ contradiction_finder  — Find conflicting studies
🔍 citation_explorer    — Find landmark and recent papers
📋 journal_club         — Critically analyze a publication
🕳️ research_gap_finder  — Identify unexplored opportunities

All agents use LLM-powered knowledge tools.
"""

from typing import Any, Callable
from .registry import AgentMeta, register_agent


# ═══════════════════════════════════════════════════════════════
# 1. Paper Summarizer
# ═══════════════════════════════════════════════════════════════

def _run_paper_summarizer(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "paper_summarizer", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "paper_summarizer_knowledge" in tools:
        try:
            result = tools["paper_summarizer_knowledge"](question=question)
            output["tool_results"]["paper_summarizer_knowledge"] = result
            output["summary"] = "Paper summarizer: synthesis complete"
        except Exception as e:
            output["tool_results"]["paper_summarizer_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Paper summarizer failed: {e}"
    output["warnings"].append("AI summary — verify key findings against original papers.")
    return output

register_agent(AgentMeta(
    name="paper_summarizer",
    description="Summarize one or many papers into key findings, methods, results, and significance. Extract structured summaries across multiple papers at once.",
    domain="literature",
    tool_names=["paper_summarizer_knowledge"],
    requires_llm=True,
    run_func=_run_paper_summarizer,
))


# ═══════════════════════════════════════════════════════════════
# 2. Evidence Synthesizer
# ═══════════════════════════════════════════════════════════════

def _run_evidence_synthesizer(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "evidence_synthesizer", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "evidence_synthesizer_knowledge" in tools:
        try:
            result = tools["evidence_synthesizer_knowledge"](question=question)
            output["tool_results"]["evidence_synthesizer_knowledge"] = result
            output["summary"] = "Evidence synthesizer: consensus built"
        except Exception as e:
            output["tool_results"]["evidence_synthesizer_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Evidence synthesizer failed: {e}"
    output["warnings"].append("Consensus is AI-assessed — verify with primary literature.")
    return output

register_agent(AgentMeta(
    name="evidence_synthesizer",
    description="Build evidence consensus from multiple studies. Weighs study quality, sample sizes, and effect directions. Identifies where the preponderance of evidence lies.",
    domain="literature",
    tool_names=["evidence_synthesizer_knowledge"],
    requires_llm=True,
    run_func=_run_evidence_synthesizer,
))


# ═══════════════════════════════════════════════════════════════
# 3. Contradiction Finder
# ═══════════════════════════════════════════════════════════════

def _run_contradiction_finder(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "contradiction_finder", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "contradiction_finder_knowledge" in tools:
        try:
            result = tools["contradiction_finder_knowledge"](question=question)
            output["tool_results"]["contradiction_finder_knowledge"] = result
            output["summary"] = "Contradiction finder: analysis complete"
        except Exception as e:
            output["tool_results"]["contradiction_finder_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Contradiction finder failed: {e}"
    output["warnings"].append("Conflicting findings may reflect methodological differences — review study designs.")
    return output

register_agent(AgentMeta(
    name="contradiction_finder",
    description="Identify studies with conflicting results on a given topic. Analyze why they disagree (methods, populations, time periods) and assess which evidence is stronger.",
    domain="literature",
    tool_names=["contradiction_finder_knowledge"],
    requires_llm=True,
    run_func=_run_contradiction_finder,
))


# ═══════════════════════════════════════════════════════════════
# 4. Citation Explorer
# ═══════════════════════════════════════════════════════════════

def _run_citation_explorer(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "citation_explorer", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "citation_explorer_knowledge" in tools:
        try:
            result = tools["citation_explorer_knowledge"](question=question)
            output["tool_results"]["citation_explorer_knowledge"] = result
            output["summary"] = "Citation explorer: landmark and recent papers identified"
        except Exception as e:
            output["tool_results"]["citation_explorer_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Citation explorer failed: {e}"
    output["warnings"].append("Citation landscape is AI-estimated — verify with Web of Science / Semantic Scholar.")
    return output

register_agent(AgentMeta(
    name="citation_explorer",
    description="Find landmark papers AND recent breakthroughs on any topic. Trace citation networks, identify seminal works, and highlight the most influential recent publications.",
    domain="literature",
    tool_names=["citation_explorer_knowledge"],
    requires_llm=True,
    run_func=_run_citation_explorer,
))


# ═══════════════════════════════════════════════════════════════
# 5. Journal Club
# ═══════════════════════════════════════════════════════════════

def _run_journal_club(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "journal_club", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "journal_club_knowledge" in tools:
        try:
            result = tools["journal_club_knowledge"](question=question)
            output["tool_results"]["journal_club_knowledge"] = result
            output["summary"] = "Journal club: critical analysis complete"
        except Exception as e:
            output["tool_results"]["journal_club_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Journal club failed: {e}"
    output["warnings"].append("Critical analysis is AI-generated — discuss with colleagues before forming conclusions.")
    return output

register_agent(AgentMeta(
    name="journal_club",
    description="Critically analyze a publication as if presenting at journal club. Evaluate hypothesis, methods, results, statistics, conclusions, and significance. Identify strengths and weaknesses.",
    domain="literature",
    tool_names=["journal_club_knowledge"],
    requires_llm=True,
    run_func=_run_journal_club,
))


# ═══════════════════════════════════════════════════════════════
# 6. Research Gap Finder
# ═══════════════════════════════════════════════════════════════

def _run_research_gap_finder(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "research_gap_finder", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "research_gap_finder_knowledge" in tools:
        try:
            result = tools["research_gap_finder_knowledge"](question=question)
            output["tool_results"]["research_gap_finder_knowledge"] = result
            output["summary"] = "Research gap finder: gaps identified"
        except Exception as e:
            output["tool_results"]["research_gap_finder_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Research gap finder failed: {e}"
    output["warnings"].append("Gap analysis is AI-generated — verify novelty with systematic literature search.")
    return output

register_agent(AgentMeta(
    name="research_gap_finder",
    description="Identify unexplored research opportunities. What important questions remain unanswered? Where are the knowledge gaps? What would a high-impact next study look like?",
    domain="literature",
    tool_names=["research_gap_finder_knowledge"],
    requires_llm=True,
    run_func=_run_research_gap_finder,
))
