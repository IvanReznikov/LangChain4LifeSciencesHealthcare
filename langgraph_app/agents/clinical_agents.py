"""
LangGraph LifeScienceBench — Clinical, Regulatory & Safety Team (3 agents).

🏥 clinical_trial_analyst — Analyze and compare clinical trials
📋 regulatory_advisor     — FDA/EMA regulatory guidance
🛡️ safety_reviewer        — Toxicology and safety assessment

All agents use LLM-powered knowledge tools.
"""

from typing import Any, Callable
from .registry import AgentMeta, register_agent


# ═══════════════════════════════════════════════════════════════
# 1. Clinical Trial Analyst
# ═══════════════════════════════════════════════════════════════

def _run_clinical_trial_analyst(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "clinical_trial_analyst", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "clinical_trial_analyst_knowledge" in tools:
        try:
            result = tools["clinical_trial_analyst_knowledge"](question=question)
            output["tool_results"]["clinical_trial_analyst_knowledge"] = result
            output["summary"] = "Clinical trial analyst: analysis complete"
        except Exception as e:
            output["tool_results"]["clinical_trial_analyst_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Clinical trial analyst failed: {e}"
    output["warnings"].append(
        "Clinical trial analysis is informational. Do NOT make treatment decisions based on AI analysis. "
        "Consult prescribing information and clinical guidelines."
    )
    return output

register_agent(AgentMeta(
    name="clinical_trial_analyst",
    description="Analyze and compare clinical trials — Phase I-III design, endpoints, statistical power, patient populations, results interpretation. Compare trial outcomes across competing therapies.",
    domain="clinical",
    tool_names=["clinical_trial_analyst_knowledge"],
    requires_llm=True,
    run_func=_run_clinical_trial_analyst,
))


# ═══════════════════════════════════════════════════════════════
# 2. Regulatory Advisor
# ═══════════════════════════════════════════════════════════════

def _run_regulatory_advisor(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "regulatory_advisor", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "regulatory_advisor_knowledge" in tools:
        try:
            result = tools["regulatory_advisor_knowledge"](question=question)
            output["tool_results"]["regulatory_advisor_knowledge"] = result
            output["summary"] = "Regulatory advisor: guidance provided"
        except Exception as e:
            output["tool_results"]["regulatory_advisor_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Regulatory advisor failed: {e}"
    output["warnings"].append(
        "Regulatory guidance is informational and may be outdated. Always consult official FDA/EMA/ICH "
        "guidance documents and regulatory affairs professionals for current requirements."
    )
    return output

register_agent(AgentMeta(
    name="regulatory_advisor",
    description="Provide FDA/EMA/ICH regulatory guidance — biomarker qualification, IND/NDA requirements, clinical trial design for registration, expedited pathways (Breakthrough, Fast Track, PRIME).",
    domain="clinical",
    tool_names=["regulatory_advisor_knowledge"],
    requires_llm=True,
    run_func=_run_regulatory_advisor,
))


# ═══════════════════════════════════════════════════════════════
# 3. Safety Reviewer
# ═══════════════════════════════════════════════════════════════

def _run_safety_reviewer(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "safety_reviewer", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "safety_reviewer_knowledge" in tools:
        try:
            result = tools["safety_reviewer_knowledge"](question=question)
            output["tool_results"]["safety_reviewer_knowledge"] = result
            output["summary"] = "Safety reviewer: assessment complete"
        except Exception as e:
            output["tool_results"]["safety_reviewer_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Safety reviewer failed: {e}"
    output["warnings"].append(
        "Toxicology assessment is AI-generated — NOT a substitute for GLP safety studies, "
        "expert toxicologist review, or regulatory safety evaluation."
    )
    return output

register_agent(AgentMeta(
    name="safety_reviewer",
    description="Review potential toxic liabilities of compounds — hERG, CYP inhibition, genotoxicity, hepatotoxicity, DILI, carcinogenicity. Identify structural alerts and recommend follow-up assays.",
    domain="clinical",
    tool_names=["safety_reviewer_knowledge"],
    requires_llm=True,
    run_func=_run_safety_reviewer,
))
