"""
LangGraph LifeScienceBench — Biology Tools.
Protein sequence QC, FASTA parsing, amino acid composition.
"""

import re
from collections import Counter
from typing import Any

from .registry import ToolMeta, register_tool


# ── Add knowledge tools import ────────────────────────────────
from .knowledge_tools import make_knowledge_tool, BIOLOGY_KNOWLEDGE_PROMPT


# ── Tool 4: Protein Sequence QC ────────────────────────────────
def _protein_sequence_qc(question: str = "", sequence: str = "", **kwargs) -> dict[str, Any]:
    """Parse and QC a protein/peptide sequence."""
    result: dict[str, Any] = {
        "tool": "protein_sequence_qc",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": [],
    }

    # Try to extract sequence from question if not provided directly
    if not sequence:
        seqs = re.findall(r'\b[ACDEFGHIKLMNPQRSTVWY]{20,}\b', question.upper())
        if not seqs:
            result["summary"] = "No protein sequence ≥20 residues found. Provide a FASTA or raw sequence."
            return result
        sequence = seqs[0]

    sequence = re.sub(r'\s+', '', sequence).upper()
    invalid_residues = set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
    composition = dict(Counter(sequence))

    result["data"] = {
        "length": len(sequence),
        "composition": composition,
        "invalid_residues": list(invalid_residues) if invalid_residues else [],
        "is_valid": len(invalid_residues) == 0,
        "aromatic_ratio": round(sum(composition.get(aa, 0) for aa in "FWY") / len(sequence), 3) if sequence else 0,
        "charged_ratio": round(sum(composition.get(aa, 0) for aa in "DEKRH") / len(sequence), 3) if sequence else 0,
    }

    if invalid_residues:
        result["warnings"].append(f"Invalid residues found: {', '.join(invalid_residues)}")
    if len(sequence) < 20:
        result["warnings"].append("Sequence too short for reliable QC (<20 residues)")

    result["status"] = "success"
    result["summary"] = f"Protein sequence QC: {len(sequence)} residues, {len(composition)} unique AAs"
    result["warnings"].append("No functional, structural, or clinical inference. Sequence validation only.")
    return result


register_tool(ToolMeta(
    name="protein_sequence_qc",
    description="Parse and QC a protein/peptide sequence — length, amino acid composition, invalid residues, aromatic/charged ratios. Provide a FASTA or raw sequence.",
    domain="biology",
    requires_input=["sequence"],
    produces="artifact",
    func=_protein_sequence_qc,
))


# ── Tool 5: FASTA/GenBank Parser ───────────────────────────────
def _fasta_parser(fasta_text: str = "", **kwargs) -> dict[str, Any]:
    """Parse multi-record FASTA text and return per-record metadata."""
    result: dict[str, Any] = {
        "tool": "fasta_parser",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": [],
    }

    if not fasta_text:
        result["summary"] = "No FASTA text provided."
        return result

    try:
        records = []
        header = None
        seq_lines = []

        for line in fasta_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header:
                    seq = ''.join(seq_lines)
                    records.append({
                        "id": header.split()[0],
                        "description": header,
                        "length": len(seq),
                        "sequence_preview": seq[:50] + ('...' if len(seq) > 50 else ''),
                    })
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(re.sub(r'\s+', '', line).upper())

        if header:
            seq = ''.join(seq_lines)
            records.append({
                "id": header.split()[0],
                "description": header,
                "length": len(seq),
                "sequence_preview": seq[:50] + ('...' if len(seq) > 50 else ''),
            })

        result["data"] = {
            "record_count": len(records),
            "records": records,
            "total_residues": sum(r["length"] for r in records),
        }
        result["status"] = "success"
        result["summary"] = f"Parsed {len(records)} FASTA record(s), {result['data']['total_residues']} total residues"
    except Exception as e:
        result["status"] = "error"
        result["summary"] = str(e)

    return result


register_tool(ToolMeta(
    name="fasta_parser",
    description="Parse multi-record FASTA text — extract record IDs, descriptions, lengths, and sequence previews.",
    domain="biology",
    requires_input=["fasta_text"],
    produces="artifact",
    func=_fasta_parser,
))


# ── Tool 6: Omics QC ───────────────────────────────────────────
def _omics_qc(csv_text: str = "", **kwargs) -> dict[str, Any]:
    """Basic QC for omics-style expression/matrix data."""
    result: dict[str, Any] = {
        "tool": "omics_qc",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": [],
    }

    if not csv_text:
        result["summary"] = "No CSV data provided for omics QC."
        return result

    try:
        import pandas as pd
        from io import StringIO

        df = pd.read_csv(StringIO(csv_text))
        numeric_cols = list(df.select_dtypes('number').columns)

        result["data"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": numeric_cols,
            "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
            "missing_pct": {k: round(v, 2) for k, v in (df.isna().sum() / len(df) * 100).to_dict().items()},
        }

        if numeric_cols:
            stats = df[numeric_cols].describe().to_dict()
            result["data"]["numeric_summary"] = {
                col: {"mean": round(stats["mean"][col], 3), "std": round(stats["std"][col], 3)}
                for col in numeric_cols
            }

        result["status"] = "success"
        result["summary"] = f"Omics QC: {len(df)} rows, {len(numeric_cols)} numeric columns"
        result["warnings"].append("Preprocessing, batch effects, and feature definitions require expert review.")
    except Exception as e:
        result["status"] = "error"
        result["summary"] = str(e)

    return result


register_tool(ToolMeta(
    name="omics_qc",
    description="Basic QC for omics expression/matrix data — missing values, numeric summaries, column profiling. Provide CSV text.",
    domain="biology",
    requires_input=["csv_text"],
    produces="artifact",
    func=_omics_qc,
))


# ── Tool: Biology Knowledge (LLM-powered, ALWAYS fires) ──────
register_tool(make_knowledge_tool(
    domain="biology",
    tool_name="biology_knowledge",
    description="Answer ANY biology question: molecular/cell biology, genetics, pathways, CRISPR, immunology, systems biology — open-ended. Always runs.",
    domain_label="Biology",
    system_prompt=BIOLOGY_KNOWLEDGE_PROMPT,
))
