"""
LangGraph LifeScienceBench — Peer Review & Grants Team (2 agents).

🔬 peer_reviewer  — Review manuscripts like a Nature/Science reviewer
💰 grant_reviewer — Evaluate funding proposals (NIH, ERC, etc.)

Both agents use LLM-powered knowledge tools.
"""

from typing import Any, Callable
from .registry import AgentMeta, register_agent


# ═══════════════════════════════════════════════════════════════
# 1. Peer Reviewer
# ═══════════════════════════════════════════════════════════════

def _run_peer_reviewer(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "peer_reviewer", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "peer_reviewer_knowledge" in tools:
        try:
            result = tools["peer_reviewer_knowledge"](question=question)
            output["tool_results"]["peer_reviewer_knowledge"] = result
            output["summary"] = "Peer reviewer: critique complete"
        except Exception as e:
            output["tool_results"]["peer_reviewer_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Peer reviewer failed: {e}"
    output["warnings"].append("AI review is a draft aid — human peer review is essential for publication decisions.")
    return output

register_agent(AgentMeta(
    name="peer_reviewer",
    description="Review a manuscript like a Nature/Science reviewer. Identify methodological weaknesses, statistical concerns, interpretation overreach, missing controls, and suggest improvements.",
    domain="review",
    tool_names=["peer_reviewer_knowledge"],
    requires_llm=True,
    run_func=_run_peer_reviewer,
))


# ═══════════════════════════════════════════════════════════════
# 2. Grant Reviewer
# ═══════════════════════════════════════════════════════════════

def _run_grant_reviewer(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "grant_reviewer", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "grant_reviewer_knowledge" in tools:
        try:
            result = tools["grant_reviewer_knowledge"](question=question)
            output["tool_results"]["grant_reviewer_knowledge"] = result
            output["summary"] = "Grant reviewer: evaluation complete"
        except Exception as e:
            output["tool_results"]["grant_reviewer_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Grant reviewer failed: {e}"
    output["warnings"].append("AI grant review is advisory — actual review panels have different criteria and expertise.")
    return output

register_agent(AgentMeta(
    name="grant_reviewer",
    description="Evaluate a funding proposal (NIH R01, ERC, Wellcome, etc.). Assess significance, innovation, approach, investigator qualifications, and environment. Identify strengths and weaknesses.",
    domain="review",
    tool_names=["grant_reviewer_knowledge"],
    requires_llm=True,
    run_func=_run_grant_reviewer,
))
