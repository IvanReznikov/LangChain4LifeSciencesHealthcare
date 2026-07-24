"""
LangGraph LifeScienceBench — Research Planner Agent.
Strategic research planning: PhD roadmaps, grant strategy, experiment prioritization,
literature gap analysis, publication strategy, career development.
Uses tools: research_planner_knowledge (LLM-powered, always fires).
"""

from typing import Any, Callable

from .registry import AgentMeta, register_agent


def _run_research_planner_agent(
    question: str,
    tools: dict[str, Callable],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute the research planner agent."""
    output: dict[str, Any] = {
        "agent": "research_planner",
        "status": "success",
        "summary": "",
        "tool_results": {},
        "warnings": [],
    }

    tool_results = {}

    if "research_planner_knowledge" in tools:
        try:
            result = tools["research_planner_knowledge"](question=question)
            tool_results["research_planner_knowledge"] = result
        except Exception as e:
            tool_results["research_planner_knowledge"] = {"status": "error", "summary": str(e)}

    output["tool_results"] = tool_results
    successes = [k for k, v in tool_results.items() if v.get("status") == "success"]

    if successes:
        output["summary"] = f"Research planner: {', '.join(successes)} succeeded"
    else:
        output["status"] = "error"
        output["summary"] = "Research planner knowledge tool failed."

    output["warnings"].append(
        "Research planning advice is strategic guidance, not a guarantee. "
        "Funding landscapes, institutional policies, and PI preferences vary. "
        "Validate all suggestions with mentors, program officers, and colleagues."
    )

    return output


register_agent(AgentMeta(
    name="research_planner",
    description="Research strategist — PhD roadmaps, grant strategy (NIH/ERC/etc.), experiment prioritization, literature gap analysis, publication strategy, career development. Like a senior PI advising on research direction. LLM-powered.",
    domain="research",
    tool_names=["research_planner_knowledge"],
    requires_llm=True,
    run_func=_run_research_planner_agent,
))
