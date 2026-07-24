"""
LangGraph LifeScienceBench — Bioinformatics Agent.
Domain expert for bioinformatics: NGS analysis, variant calling, pathway enrichment,
genome assembly, phylogenetics, structural bioinformatics.
Uses tools: bioinformatics_knowledge (LLM-powered, always fires).
"""

from typing import Any, Callable

from .registry import AgentMeta, register_agent


def _run_bioinformatics_agent(
    question: str,
    tools: dict[str, Callable],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute the bioinformatics agent."""
    output: dict[str, Any] = {
        "agent": "bioinformatics",
        "status": "success",
        "summary": "",
        "tool_results": {},
        "warnings": [],
    }

    tool_results = {}

    if "bioinformatics_knowledge" in tools:
        try:
            result = tools["bioinformatics_knowledge"](question=question)
            tool_results["bioinformatics_knowledge"] = result
        except Exception as e:
            tool_results["bioinformatics_knowledge"] = {"status": "error", "summary": str(e)}

    output["tool_results"] = tool_results
    successes = [k for k, v in tool_results.items() if v.get("status") == "success"]

    if successes:
        output["summary"] = f"Bioinformatics agent: {', '.join(successes)} succeeded"
    else:
        output["status"] = "error"
        output["summary"] = "Bioinformatics knowledge tool failed."

    output["warnings"].append(
        "Bioinformatics analysis is computational. Validate results with experimental data "
        "and consult domain experts for clinical interpretation."
    )

    return output


register_agent(AgentMeta(
    name="bioinformatics",
    description="Bioinformatics expert — NGS analysis, variant calling, pathway enrichment, genome assembly, phylogenetics, structural bioinformatics, tool/pipeline recommendations. LLM-powered for ANY bioinformatics question.",
    domain="bioinformatics",
    tool_names=["bioinformatics_knowledge"],
    requires_llm=True,
    run_func=_run_bioinformatics_agent,
))
