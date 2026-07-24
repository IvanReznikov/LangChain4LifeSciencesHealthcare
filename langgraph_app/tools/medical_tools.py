"""
LangGraph LifeScienceBench — Medical/Healthcare Tools.
FHIR QC, PICO extraction, data quality checks.
CRITICAL: All medical tools are research-only. Clinical actions are BLOCKED.
"""

import re
import json
from typing import Any

from .registry import ToolMeta, register_tool
from .knowledge_tools import make_knowledge_tool, MEDICAL_KNOWLEDGE_PROMPT

# ── Clinical action keywords that trigger BLOCK ────────────────
_CLINICAL_BLOCK_KEYWORDS = [
    'diagnos', 'treat', 'dosage', 'prescrib', 'triage',
    'final billing', 'icd code', 'ehr write', 'patient message',
    'surgery', 'prognosis', 'referral', 'discharge', 'admit',
]


def _check_clinical_block(question: str) -> dict[str, Any] | None:
    """Check if question triggers clinical block. Returns block artifact or None."""
    q_lower = question.lower()
    for kw in _CLINICAL_BLOCK_KEYWORDS:
        if kw in q_lower:
            return {
                "tool": "medical_safety_gate",
                "status": "blocked",
                "summary": f"High-impact clinical action '{kw}' is blocked. Research support only.",
                "data": {"blocked_keyword": kw},
                "warnings": [
                    "This is a research tool, not clinical decision support.",
                    "Do not diagnose, treat, prescribe, or make clinical decisions with this tool."
                ],
            }
    return None


# ── Tool 7: Medical Safety Gate ────────────────────────────────
def _medical_safety_gate(question: str = "", **kwargs) -> dict[str, Any]:
    """Check if a medical question triggers a clinical-action block."""
    block = _check_clinical_block(question)
    if block:
        return block
    return {
        "tool": "medical_safety_gate",
        "status": "success",
        "summary": "Question passed safety gate — research support only.",
        "data": {"passed": True},
        "warnings": [
            "Research support only: public/synthetic/de-identified data; qualified review required."
        ],
    }


register_tool(ToolMeta(
    name="medical_safety_gate",
    description="Safety gate that blocks high-impact clinical actions (diagnose, treat, prescribe, etc.). Always runs first in medical workflows.",
    domain="medical",
    requires_input=["question"],
    produces="artifact",
    is_dangerous=False,  # This is the safety layer itself
    func=_medical_safety_gate,
))


# ── Tool 8: FHIR QC ────────────────────────────────────────────
def _fhir_qc(fhir_text: str = "", **kwargs) -> dict[str, Any]:
    """Parse and inventory FHIR-like JSON resources."""
    result: dict[str, Any] = {
        "tool": "fhir_qc",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": [],
    }

    if not fhir_text:
        result["summary"] = "No FHIR JSON provided."
        return result

    try:
        data = json.loads(fhir_text)
        if isinstance(data, dict) and data.get('resourceType') == 'Bundle':
            resources = [e.get('resource', {}) for e in data.get('entry', [])]
        else:
            resources = [data]

        types = {}
        for r in resources:
            rt = r.get('resourceType', 'Unknown')
            types[rt] = types.get(rt, 0) + 1

        result["data"] = {
            "resource_count": len(resources),
            "resource_types": types,
        }
        result["status"] = "review_required"
        result["summary"] = f"FHIR resource inventory: {len(resources)} resources, {len(types)} types"
        result["warnings"].append("Not a full FHIR validation or clinical timeline. Qualified review required.")
    except json.JSONDecodeError:
        result["status"] = "error"
        result["summary"] = "Invalid JSON in FHIR text."
    except Exception as e:
        result["status"] = "error"
        result["summary"] = str(e)

    return result


register_tool(ToolMeta(
    name="fhir_qc",
    description="Parse FHIR-like JSON (Bundle or single Resource) and produce a resource-type inventory. Research only — not full FHIR validation.",
    domain="medical",
    requires_input=["fhir_text"],
    produces="artifact",
    func=_fhir_qc,
))


# ── Tool 9: PICO Extraction ────────────────────────────────────
def _pico_extraction(question: str = "", **kwargs) -> dict[str, Any]:
    """Extract PICO (Population, Intervention, Comparison, Outcome) elements from a clinical question."""
    result: dict[str, Any] = {
        "tool": "pico_extraction",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": [],
    }

    if not question:
        result["summary"] = "No clinical question provided for PICO extraction."
        return result

    # Rule-based PICO extraction (lightweight, no LLM required)
    pico = {
        "population": None,
        "intervention": None,
        "comparison": None,
        "outcome": None,
    }

    # Population patterns
    pop_patterns = [
        r'(?:in|among|for)\s+(patients?\s+(?:with|of)\s+[^,.;]+)',
        r'(?:adult|pediatric|elderly|neonatal)\s+patients?\s+(?:with\s+)?([^,.;]+)',
    ]
    for pat in pop_patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            pico["population"] = m.group(1).strip()
            break

    # Intervention patterns
    int_patterns = [
        r'(?:treatment|therapy|intervention|drug|medication)\s+(?:of|with|using)\s+([^,.;]+)',
        r'(?:treated\s+(?:with|using)\s+([^,.;]+))',
    ]
    for pat in int_patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            pico["intervention"] = m.group(1).strip()
            break

    # Comparison patterns
    comp_patterns = [
        r'(?:compared?\s+(?:to|with|versus|vs\.?)\s+([^,.;]+))',
        r'(?:versus|vs\.?)\s+([^,.;]+)',
    ]
    for pat in comp_patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            pico["comparison"] = m.group(1).strip()
            break

    # Outcome patterns
    out_patterns = [
        r'(?:outcome|endpoint|measure|assess|evaluate)\s+(?:of\s+)?([^,.;]+)',
        r'(?:mortality|survival|efficacy|safety|quality\s+of\s+life)',
    ]
    for pat in out_patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            pico["outcome"] = m.group(0).strip()
            break

    found_any = any(v is not None for v in pico.values())
    result["data"] = pico
    result["status"] = "success" if found_any else "abstain"
    result["summary"] = f"PICO extraction: {sum(1 for v in pico.values() if v)}/4 elements found"
    if not found_any:
        result["warnings"].append("Could not extract PICO elements. Rephrase as a structured clinical question.")

    return result


register_tool(ToolMeta(
    name="pico_extraction",
    description="Extract PICO (Population, Intervention, Comparison, Outcome) elements from a clinical research question. Rule-based, no LLM.",
    domain="medical",
    requires_input=["question"],
    produces="artifact",
    func=_pico_extraction,
))


# ── Tool: Medical Knowledge (LLM-powered, ALWAYS fires) ──────
register_tool(make_knowledge_tool(
    domain="medical",
    tool_name="medical_knowledge",
    description="Answer ANY medical research question: disease mechanisms, clinical evidence, guidelines, pharmacology, epidemiology — open-ended. Always runs.",
    domain_label="Medical Research",
    system_prompt=MEDICAL_KNOWLEDGE_PROMPT,
))
