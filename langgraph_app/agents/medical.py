"""
LangGraph LifeScienceBench — Medical/Healthcare Agent.
Research-only domain expert with mandatory safety gate.
Clinical actions (diagnose, treat, prescribe, etc.) are BLOCKED.
Uses tools: medical_safety_gate, fhir_qc, pico_extraction.
"""

from typing import Any, Callable

from .registry import AgentMeta, register_agent


def _run_medical_agent(
    question: str,
    tools: dict[str, Callable],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute the medical agent. Safety gate ALWAYS runs first."""
    output: dict[str, Any] = {
        "agent": "medical",
        "status": "success",
        "summary": "",
        "tool_results": {},
        "warnings": [],
    }

    tool_results = {}

    # ── ALWAYS run safety gate first ──
    if "medical_safety_gate" in tools:
        try:
            gate_result = tools["medical_safety_gate"](question=question)
            tool_results["medical_safety_gate"] = gate_result
            if gate_result.get("status") == "blocked":
                output["status"] = "blocked"
                output["summary"] = f"BLOCKED: {gate_result.get('summary', '')}"
                output["tool_results"] = tool_results
                output["warnings"] = gate_result.get("warnings", [])
                return output
        except Exception as e:
            tool_results["medical_safety_gate"] = {"status": "error", "summary": str(e)}

    # ── PICO extraction for clinical research questions ──
    if any(kw in question.lower() for kw in ['pico', 'clinical question', 'research question', 'evidence']):
        if "pico_extraction" in tools:
            try:
                result = tools["pico_extraction"](question=question)
                tool_results["pico_extraction"] = result
            except Exception as e:
                tool_results["pico_extraction"] = {"status": "error", "summary": str(e)}

    # ── FHIR QC if FHIR text provided ──
    fhir_text = state.get("tool_inputs", {}).get("fhir_text", "")
    if fhir_text and "fhir_qc" in tools:
        try:
            result = tools["fhir_qc"](fhir_text=fhir_text)
            tool_results["fhir_qc"] = result
        except Exception as e:
            tool_results["fhir_qc"] = {"status": "error", "summary": str(e)}

    output["tool_results"] = tool_results

    successes = [k for k, v in tool_results.items() if v.get("status") in ("success", "review_required")]
    errors = [k for k, v in tool_results.items() if v.get("status") == "error"]

    if successes:
        output["summary"] = f"Medical agent ran {len(tool_results)} tool(s): {', '.join(successes)}"
    elif errors:
        output["status"] = "error"
        output["summary"] = f"Medical tools failed: {', '.join(errors)}"
    else:
        output["summary"] = "Medical safety gate passed. No specific medical tools matched — providing research-only context."

    output["warnings"].append(
        "Medical research support only. Public/synthetic/deidentified material only. "
        "Do not diagnose, treat, triage, prescribe, dose, code finally, write EHRs, or message patients. "
        "Qualified review required."
    )

    return output


register_agent(AgentMeta(
    name="medical",
    description="Medical research support agent — safety-gated (blocks clinical actions), FHIR resource inventory, PICO element extraction. Research only.",
    domain="medical",
    tool_names=["medical_safety_gate", "fhir_qc", "pico_extraction", "medical_knowledge"],
    requires_llm=False,
    is_safe_for_auto=False,  # Always requires user awareness
    run_func=_run_medical_agent,
))
