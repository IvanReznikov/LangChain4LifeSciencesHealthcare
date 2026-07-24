"""
LangGraph LifeScienceBench — Chemistry Agent.
Domain expert for molecular analysis, cheminformatics, and assay curation.
Uses tools: molecular_identity, chemical_similarity, assay_curation.
"""

from typing import Any, Callable

from .registry import AgentMeta, register_agent


def _run_chemistry_agent(
    question: str,
    tools: dict[str, Callable],
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute the chemistry agent with available chemistry tools.
    Returns agent output dict with status, summary, and tool results.
    """
    output: dict[str, Any] = {
        "agent": "chemistry",
        "status": "success",
        "summary": "",
        "tool_results": {},
        "warnings": [],
    }

    tool_results = {}

    # ── Run all available tools — each tool self-detects applicability ──
    # Tools return 'abstain' when they can't process the input.
    q_lower = question.lower()

    # 1. Molecular identity — always try; tool detects SMILES internally
    if "molecular_identity" in tools:
        try:
            result = tools["molecular_identity"](question=question)
            tool_results["molecular_identity"] = result
        except Exception as e:
            tool_results["molecular_identity"] = {"status": "error", "summary": str(e)}

    # 2. Chemical similarity — try if question has multiple structural tokens
    if "chemical_similarity" in tools:
        try:
            result = tools["chemical_similarity"](question=question)
            if result.get("status") != "abstain":
                tool_results["chemical_similarity"] = result
        except Exception as e:
            tool_results["chemical_similarity"] = {"status": "error", "summary": str(e)}

    # 3. Assay curation — if CSV data provided
    csv_text = state.get("tool_inputs", {}).get("csv_text", "")
    if csv_text and "assay_curation" in tools:
        try:
            result = tools["assay_curation"](csv_text=csv_text)
            tool_results["assay_curation"] = result
        except Exception as e:
            tool_results["assay_curation"] = {"status": "error", "summary": str(e)}

    output["tool_results"] = tool_results

    # ── Build summary ──
    successes = [k for k, v in tool_results.items() if v.get("status") == "success"]
    errors = [k for k, v in tool_results.items() if v.get("status") == "error"]

    if successes:
        output["summary"] = f"Chemistry agent ran {len(tool_results)} tool(s): {', '.join(successes)} succeeded"
    elif errors:
        output["status"] = "error"
        output["summary"] = f"Chemistry tools failed: {', '.join(errors)}"
    else:
        output["status"] = "abstain"
        output["summary"] = "No chemistry tools matched the question. Try providing a SMILES string or a more specific chemistry query."

    output["warnings"].append(
        "Chemical tools compute molecular properties only. No inference of activity, toxicity, synthesis feasibility, or clinical suitability."
    )

    return output


register_agent(AgentMeta(
    name="chemistry",
    description="Chemistry domain expert — molecular identity (MW, LogP, TPSA, formula), structural similarity (Tanimoto), and assay curation. Provide SMILES: <structure>.",
    domain="chemistry",
    tool_names=["molecular_identity", "chemical_similarity", "assay_curation", "chemistry_knowledge",
                "retrosynthesis_planner", "spectroscopy_interpreter", "reaction_predictor"],
    requires_llm=False,
    run_func=_run_chemistry_agent,
))
