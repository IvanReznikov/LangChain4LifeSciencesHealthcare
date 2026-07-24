"""
LangGraph LifeScienceBench — Drug Discovery Agent.
Domain expert for drug discovery questions: target ID, hit discovery, lead optimization,
ADMET, medicinal chemistry, computational drug design, clinical development strategy.
Uses tools: drug_discovery_knowledge (LLM-powered, always fires).
"""

from typing import Any, Callable

from .registry import AgentMeta, register_agent


def _run_drug_discovery_agent(
    question: str,
    tools: dict[str, Callable],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute the drug discovery agent with available tools."""
    output: dict[str, Any] = {
        "agent": "drug_discovery",
        "status": "success",
        "summary": "",
        "tool_results": {},
        "warnings": [],
    }

    tool_results = {}

    # ── Drug Discovery Knowledge (LLM-powered, ALWAYS fires) ──
    if "drug_discovery_knowledge" in tools:
        try:
            result = tools["drug_discovery_knowledge"](question=question)
            tool_results["drug_discovery_knowledge"] = result
        except Exception as e:
            tool_results["drug_discovery_knowledge"] = {"status": "error", "summary": str(e)}

    output["tool_results"] = tool_results
    successes = [k for k, v in tool_results.items() if v.get("status") == "success"]
    errors = [k for k, v in tool_results.items() if v.get("status") == "error"]

    if successes:
        output["summary"] = f"Drug discovery agent: {', '.join(successes)} succeeded"
    elif errors:
        output["status"] = "error"
        output["summary"] = f"Drug discovery tools failed: {', '.join(errors)}"

    output["warnings"].append(
        "Drug discovery analysis is computational/AI-generated. All predictions require "
        "experimental validation. Patent and regulatory guidance is informational only."
    )

    return output


register_agent(AgentMeta(
    name="drug_discovery",
    description="Drug discovery expert — target validation, hit finding, lead optimization, ADMET, medicinal chemistry, computational drug design, clinical development strategy. LLM-powered for ANY drug discovery question.",
    domain="drug_discovery",
    tool_names=["drug_discovery_knowledge"],
    requires_llm=True,
    run_func=_run_drug_discovery_agent,
))
