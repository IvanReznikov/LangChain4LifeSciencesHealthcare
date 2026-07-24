"""
LangGraph LifeScienceBench — IP & Competitive Intelligence Team (2 agents).

📜 patent_search            — Patent landscape analysis
🏢 competitive_intelligence  — Pharma pipeline and competitor analysis

Both agents use LLM-powered knowledge tools.
"""

from typing import Any, Callable
from .registry import AgentMeta, register_agent


# ═══════════════════════════════════════════════════════════════
# 1. Patent Search
# ═══════════════════════════════════════════════════════════════

def _run_patent_search(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "patent_search", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "patent_search_knowledge" in tools:
        try:
            result = tools["patent_search_knowledge"](question=question)
            output["tool_results"]["patent_search_knowledge"] = result
            output["summary"] = "Patent search: landscape analysis complete"
        except Exception as e:
            output["tool_results"]["patent_search_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Patent search failed: {e}"
    output["warnings"].append(
        "Patent landscape is AI-estimated from training data — NOT a freedom-to-operate analysis. "
        "Always conduct formal patent searches with legal counsel for FTO, infringement, or filing decisions."
    )
    return output

register_agent(AgentMeta(
    name="patent_search",
    description="Analyze the patent landscape — what patents exist on a target, compound class, or technology? Identify key assignees, filing trends, composition-of-matter vs. method claims, and potential white space.",
    domain="ip",
    tool_names=["patent_search_knowledge"],
    requires_llm=True,
    run_func=_run_patent_search,
))


# ═══════════════════════════════════════════════════════════════
# 2. Competitive Intelligence
# ═══════════════════════════════════════════════════════════════

def _run_competitive_intelligence(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "competitive_intelligence", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "competitive_intelligence_knowledge" in tools:
        try:
            result = tools["competitive_intelligence_knowledge"](question=question)
            output["tool_results"]["competitive_intelligence_knowledge"] = result
            output["summary"] = "Competitive intelligence: analysis complete"
        except Exception as e:
            output["tool_results"]["competitive_intelligence_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Competitive intelligence failed: {e}"
    output["warnings"].append(
        "Competitive intelligence is AI-estimated from public data — pipelines change rapidly. "
        "Verify with Cortellis, Pharmaprojects, ClinicalTrials.gov, and company filings."
    )
    return output

register_agent(AgentMeta(
    name="competitive_intelligence",
    description="Analyze pharmaceutical pipelines and competitors — compare oncology/immunology/CNS pipelines of major pharma, identify emerging biotech players, track deal flow and M&A trends.",
    domain="ip",
    tool_names=["competitive_intelligence_knowledge"],
    requires_llm=True,
    run_func=_run_competitive_intelligence,
))
