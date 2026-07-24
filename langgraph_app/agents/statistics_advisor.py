"""
LangGraph LifeScienceBench — Statistics Advisor Agent.
Biostatistics and data science advisor: study design, test selection,
power analysis, regression modeling, meta-analysis, ML evaluation.
Uses tools: statistics_knowledge (LLM-powered, always fires).
"""

from typing import Any, Callable

from .registry import AgentMeta, register_agent


def _run_statistics_advisor_agent(
    question: str,
    tools: dict[str, Callable],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute the statistics advisor agent."""
    output: dict[str, Any] = {
        "agent": "statistics_advisor",
        "status": "success",
        "summary": "",
        "tool_results": {},
        "warnings": [],
    }

    tool_results = {}

    if "statistics_knowledge" in tools:
        try:
            result = tools["statistics_knowledge"](question=question)
            tool_results["statistics_knowledge"] = result
        except Exception as e:
            tool_results["statistics_knowledge"] = {"status": "error", "summary": str(e)}

    output["tool_results"] = tool_results
    successes = [k for k, v in tool_results.items() if v.get("status") == "success"]

    if successes:
        output["summary"] = f"Statistics advisor: {', '.join(successes)} succeeded"
    else:
        output["status"] = "error"
        output["summary"] = "Statistics knowledge tool failed."

    output["warnings"].append(
        "Statistical advice is informational only. Consult a biostatistician for study design, "
        "analysis plan, and interpretation. Statistical recommendations may have assumptions "
        "that need verification with your specific data."
    )

    return output


register_agent(AgentMeta(
    name="statistics_advisor",
    description="Biostatistics advisor — study design, test selection, power analysis, regression modeling (linear, logistic, Cox, mixed-effects), multiple testing correction, survival analysis, Bayesian methods, meta-analysis, ML evaluation. LLM-powered. Consult a biostatistician for final decisions.",
    domain="statistics",
    tool_names=["statistics_knowledge"],
    requires_llm=True,
    run_func=_run_statistics_advisor_agent,
))
