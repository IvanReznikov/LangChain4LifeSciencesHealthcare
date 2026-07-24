"""
LangGraph LifeScienceBench — Scientific Writer Agent.
Drafts scientific documents: abstracts, introductions, methods sections,
grant proposals, review articles, figure legends.
Uses tools: scientific_writer_knowledge (LLM-powered, always fires).
"""

from typing import Any, Callable

from .registry import AgentMeta, register_agent


def _run_scientific_writer_agent(
    question: str,
    tools: dict[str, Callable],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute the scientific writer agent."""
    output: dict[str, Any] = {
        "agent": "scientific_writer",
        "status": "success",
        "summary": "",
        "tool_results": {},
        "warnings": [],
    }

    tool_results = {}

    if "scientific_writer_knowledge" in tools:
        try:
            result = tools["scientific_writer_knowledge"](question=question)
            tool_results["scientific_writer_knowledge"] = result
        except Exception as e:
            tool_results["scientific_writer_knowledge"] = {"status": "error", "summary": str(e)}

    output["tool_results"] = tool_results
    successes = [k for k, v in tool_results.items() if v.get("status") == "success"]

    if successes:
        output["summary"] = f"Scientific writer: {', '.join(successes)} succeeded"
    else:
        output["status"] = "error"
        output["summary"] = "Scientific writer knowledge tool failed."

    output["warnings"].append(
        "AI-generated scientific writing is a draft aid. All content must be verified, "
        "edited, and approved by domain experts. Citations may be hallucinated — verify every one. "
        "Follow journal-specific guidelines and ICMJE authorship criteria."
    )

    return output


register_agent(AgentMeta(
    name="scientific_writer",
    description="Scientific writer — draft abstracts, introductions, methods, results, discussions, grant proposals, review articles, figure legends. LLM-powered. All output is draft-level; human review required.",
    domain="writing",
    tool_names=["scientific_writer_knowledge"],
    requires_llm=True,
    run_func=_run_scientific_writer_agent,
))
