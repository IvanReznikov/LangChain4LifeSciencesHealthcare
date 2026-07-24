"""
LangGraph LifeScienceBench — Discussion Panel Agent.
Multi-perspective scientific review board.
Simulates independent reviewer roles (domain scientist, methods/statistics critic,
data-quality reviewer) to produce a structured review artifact.
Uses LLM to simulate reviewer perspectives.
"""

from typing import Any, Callable

from .registry import AgentMeta, register_agent

# ── Reviewer role definitions ──────────────────────────────────
REVIEWER_ROLES = [
    {
        "role": "domain_scientist",
        "label": "Domain Scientist",
        "focus": "Scientific plausibility, domain knowledge consistency, biological/chemical/clinical relevance of findings.",
    },
    {
        "role": "methods_critic",
        "label": "Methods & Statistics Critic",
        "focus": "Methodological rigor, statistical assumptions, effect sizes, confidence intervals, reproducibility concerns.",
    },
    {
        "role": "data_quality_reviewer",
        "label": "Data Quality Reviewer",
        "focus": "Data provenance, missing values, batch effects, unit consistency, measurement error, selection bias.",
    },
]


def _run_discussion_agent(
    question: str,
    tools: dict[str, Callable],
    state: dict[str, Any],
    llm: Any = None,
) -> dict[str, Any]:
    """
    Execute the discussion panel agent.
    Simulates a multi-perspective review board using LLM role-play.
    """
    output: dict[str, Any] = {
        "agent": "discussion",
        "status": "success",
        "summary": "",
        "reviewers": {},
        "agreement": "",
        "dissent": "",
        "next_actions": [],
        "warnings": [],
    }

    # ── Gather context from other agents ──
    agent_outputs = state.get("agent_outputs", {})
    synthesis = state.get("synthesis", "")
    evidence = state.get("evidence", [])

    context_parts = [f"RESEARCH QUESTION: {question}"]

    if evidence:
        context_parts.append("\nEVIDENCE:")
        for e in evidence[:5]:
            context_parts.append(f"- [{e.get('citation', '?')}] {e.get('excerpt', '')[:200]}")

    for agent_name, agent_out in agent_outputs.items():
        if agent_name == "discussion":
            continue
        context_parts.append(f"\n{agent_name.upper()} AGENT OUTPUT:")
        context_parts.append(str(agent_out.get("summary", "")))
        for tool_name, tool_result in agent_out.get("tool_results", {}).items():
            if tool_result.get("status") == "success":
                context_parts.append(f"  {tool_name}: {tool_result.get('summary', '')}")

    if synthesis:
        context_parts.append(f"\nSYNTHESIS: {synthesis}")

    context = "\n".join(context_parts)

    # ── Run each reviewer role ──
    if llm:
        for role_def in REVIEWER_ROLES:
            try:
                prompt = f"""You are a {role_def['label']} on an independent scientific review board. 
Your focus: {role_def['focus']}

{context}

As the {role_def['label']}, provide your independent assessment in 2-4 sentences. Structure your response as:
FINDINGS: <what the evidence supports or doesn't support for your domain>
CONCERNS: <specific methodological or data concerns you see>
RECOMMENDATION: <your recommendation for next steps>

CRITICAL RULES:
- You are an independent reviewer — do not defer to or echo other roles.
- State assumptions and missing evidence explicitly.
- This is a research facilitation artifact, not clinical or expert authority.
- If evidence is insufficient, say so clearly — do not invent findings."""

                from langchain_core.messages import HumanMessage
                response = llm.invoke([HumanMessage(content=prompt)])
                response_text = response.content if hasattr(response, 'content') else str(response)

                # Parse structured response
                findings = ""
                concerns = ""
                recommendation = ""

                for line in response_text.split('\n'):
                    line = line.strip()
                    if line.startswith('FINDINGS:'):
                        findings = line.replace('FINDINGS:', '').strip()
                    elif line.startswith('CONCERNS:'):
                        concerns = line.replace('CONCERNS:', '').strip()
                    elif line.startswith('RECOMMENDATION:'):
                        recommendation = line.replace('RECOMMENDATION:', '').strip()

                output["reviewers"][role_def["role"]] = {
                    "label": role_def["label"],
                    "findings": findings or response_text[:200],
                    "concerns": concerns,
                    "recommendation": recommendation,
                }
            except Exception as e:
                output["reviewers"][role_def["role"]] = {
                    "label": role_def["label"],
                    "findings": f"Reviewer unavailable: {str(e)}",
                    "concerns": "",
                    "recommendation": "Manual review required.",
                }
    else:
        # No LLM — provide structured template
        for role_def in REVIEWER_ROLES:
            output["reviewers"][role_def["role"]] = {
                "label": role_def["label"],
                "findings": f"LLM not available — {role_def['label']} assessment deferred to human review.",
                "concerns": "Review the evidence and agent outputs manually.",
                "recommendation": "Enable LLM for automated review board simulation.",
            }

    # ── Build agreement/dissent summary ──
    output["agreement"] = (
        "No consensus is asserted automatically. "
        "Each reviewer independently assessed the evidence from their domain perspective."
    )
    output["dissent"] = (
        "Each role must identify assumptions and missing evidence. "
        "Disagreement between reviewers is expected and valuable — "
        "it highlights areas requiring deeper investigation."
    )
    output["next_actions"] = [
        "Validate primary sources cited in evidence",
        "Review units, metadata, and experimental design",
        "Define a minimal next experiment or analysis to resolve gaps",
        "Confirm that all findings are supported by retrievable evidence",
    ]

    reviewer_count = len(output["reviewers"])
    output["summary"] = f"Discussion panel: {reviewer_count} independent reviewers assessed the evidence"

    output["warnings"].append(
        "This is a structured facilitation artifact, not simulated expert authority. "
        "Reviewer outputs are LLM-generated and require human verification."
    )

    return output


register_agent(AgentMeta(
    name="discussion",
    description="Multi-perspective scientific review board — 3 independent reviewers (domain scientist, methods critic, data-quality reviewer) assess findings, identify gaps, and recommend next steps. Requires LLM.",
    domain="discussion",
    tool_names=[],
    requires_llm=True,
    run_func=_run_discussion_agent,
))
