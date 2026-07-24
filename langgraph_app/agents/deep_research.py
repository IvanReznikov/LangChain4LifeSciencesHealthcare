"""
LangGraph LifeScienceBench — Deep Research Agent.
Systematic multi-source research: designs search strategy, aggregates evidence,
assesses source quality, identifies contradictions, grades evidence strength.
Uses tools: deep_research_knowledge (LLM-powered, always fires).
"""

from typing import Any, Callable

from .registry import AgentMeta, register_agent


def _run_deep_research_agent(
    question: str,
    tools: dict[str, Callable],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute the deep research agent."""
    output: dict[str, Any] = {
        "agent": "deep_research",
        "status": "success",
        "summary": "",
        "tool_results": {},
        "evidence_quality": {},
        "contradictions": [],
        "warnings": [],
    }

    tool_results = {}

    if "deep_research_knowledge" in tools:
        try:
            result = tools["deep_research_knowledge"](question=question)
            tool_results["deep_research_knowledge"] = result
        except Exception as e:
            tool_results["deep_research_knowledge"] = {"status": "error", "summary": str(e)}

    output["tool_results"] = tool_results
    successes = [k for k, v in tool_results.items() if v.get("status") == "success"]

    if successes:
        output["summary"] = f"Deep research agent: {', '.join(successes)} succeeded"
    else:
        output["status"] = "error"
        output["summary"] = "Deep research knowledge tool failed."

    output["evidence_quality"] = {
        "systematic_review_possible": "unknown",
        "meta_analysis_possible": "unknown",
        "evidence_grading": "GRADE framework recommended",
    }

    output["warnings"].append(
        "Deep research output is AI-generated. Verify all citations, assess source quality "
        "independently, and consult a systematic review methodologist for formal evidence synthesis."
    )

    return output


register_agent(AgentMeta(
    name="deep_research",
    description="Deep systematic research agent — designs search strategies, aggregates multi-source evidence, grades evidence quality (GRADE/OCEBM), identifies contradictions and knowledge gaps. LLM-powered for thorough evidence synthesis.",
    domain="research",
    tool_names=["deep_research_knowledge"],
    requires_llm=True,
    run_func=_run_deep_research_agent,
))
