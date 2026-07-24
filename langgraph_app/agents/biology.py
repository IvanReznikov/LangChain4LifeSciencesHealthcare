"""
LangGraph LifeScienceBench — Biology Agent.
Domain expert for protein/peptide sequence QC, FASTA parsing, and omics analysis.
Uses tools: protein_sequence_qc, fasta_parser, omics_qc.
"""

from typing import Any, Callable

from .registry import AgentMeta, register_agent


def _run_biology_agent(
    question: str,
    tools: dict[str, Callable],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute the biology agent with available biology tools."""
    output: dict[str, Any] = {
        "agent": "biology",
        "status": "success",
        "summary": "",
        "tool_results": {},
        "warnings": [],
    }

    tool_results = {}
    q_lower = question.lower()

    # ── Run all available tools — each tool self-detects applicability ──
    # Tools use regex/parsing internally and return 'abstain' if input doesn't match.

    # 1. Protein sequence QC — always try; tool detects sequences via regex
    if "protein_sequence_qc" in tools:
        try:
            result = tools["protein_sequence_qc"](question=question)
            tool_results["protein_sequence_qc"] = result
        except Exception as e:
            tool_results["protein_sequence_qc"] = {"status": "error", "summary": str(e)}

    # 2. FASTA parser — if FASTA text provided (look for > header or raw multi-line)
    fasta_text = state.get("tool_inputs", {}).get("fasta_text", "")
    if fasta_text and "fasta_parser" in tools:
        try:
            result = tools["fasta_parser"](fasta_text=fasta_text)
            tool_results["fasta_parser"] = result
        except Exception as e:
            tool_results["fasta_parser"] = {"status": "error", "summary": str(e)}

    # 3. Omics QC — if CSV data provided
    csv_text = state.get("tool_inputs", {}).get("csv_text", "")
    if csv_text and "omics_qc" in tools:
        try:
            result = tools["omics_qc"](csv_text=csv_text)
            tool_results["omics_qc"] = result
        except Exception as e:
            tool_results["omics_qc"] = {"status": "error", "summary": str(e)}

    output["tool_results"] = tool_results

    successes = [k for k, v in tool_results.items() if v.get("status") == "success"]
    errors = [k for k, v in tool_results.items() if v.get("status") == "error"]

    if successes:
        output["summary"] = f"Biology agent ran {len(tool_results)} tool(s): {', '.join(successes)} succeeded"
    elif errors:
        output["status"] = "error"
        output["summary"] = f"Biology tools failed: {', '.join(errors)}"
    else:
        output["status"] = "abstain"
        output["summary"] = "No biology tools matched. Try providing a protein/peptide sequence (≥20 residues), FASTA text, or omics CSV data."

    output["warnings"].append(
        "Biological tools provide sequence validation and composition only. No functional, structural, or clinical inference."
    )

    return output


register_agent(AgentMeta(
    name="biology",
    description="Biology domain expert — protein/peptide sequence QC, FASTA record parsing, and omics expression data profiling. Provide sequences or CSV data.",
    domain="biology",
    tool_names=["protein_sequence_qc", "fasta_parser", "omics_qc", "biology_knowledge"],
    requires_llm=False,
    run_func=_run_biology_agent,
))
