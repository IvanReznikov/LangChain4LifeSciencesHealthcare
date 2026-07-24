"""
LangGraph LifeScienceBench — Chemistry Tools.
RDKit molecular analysis, SMILES parsing, property calculation.
Uses the existing bench/tools/domains.py chemistry function as foundation.
"""

import re
from collections import Counter
from io import StringIO
from typing import Any, Optional

import pandas as pd

from .registry import ToolMeta, register_tool
from .knowledge_tools import make_knowledge_tool, CHEMISTRY_KNOWLEDGE_PROMPT


# ── Tool 1: Molecular Identity ─────────────────────────────────
def _molecular_identity(question: str, **kwargs) -> dict[str, Any]:
    """Extract SMILES from question and compute RDKit descriptors."""
    result: dict[str, Any] = {
        "tool": "molecular_identity",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": [],
    }

    m = re.search(r'SMILES\s*[:=]\s*([^\s]+)', question, re.IGNORECASE)

    smiles_candidate = m.group(1) if m else None

    if not smiles_candidate:
        result["summary"] = "No SMILES found in question. Provide as: SMILES: <structure>"
        result["status"] = "abstain"
        return result

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, inchi, Crippen
        from rdkit import RDLogger
        RDLogger.logger().setLevel(RDLogger.ERROR)  # Suppress parse warnings

        mol = Chem.MolFromSmiles(smiles_candidate)
        if mol is None:
            result["summary"] = f"Invalid SMILES: {smiles_candidate}"
            result["status"] = "error"
            return result

        result["data"] = {
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
            "inchikey": inchi.MolToInchiKey(mol),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "mw": round(Descriptors.MolWt(mol), 3),
            "logp": round(Crippen.MolLogP(mol), 3),
            "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 3),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
        }
        result["status"] = "success"
        result["summary"] = f"RDKit molecular identity computed for {smiles_candidate}"
        result["warnings"].append(
            "Not evidence of activity, toxicity, synthesis feasibility, or clinical suitability."
        )
    except ImportError:
        result["status"] = "error"
        result["summary"] = "RDKit not installed. Install: pip install rdkit-pypi"
    except Exception as e:
        result["status"] = "error"
        result["summary"] = str(e)

    return result


register_tool(ToolMeta(
    name="molecular_identity",
    description="Compute RDKit molecular descriptors from a SMILES string (MW, LogP, TPSA, HBD/HBA, formula, InChIKey). Provide SMILES: <structure> in your question.",
    domain="chemistry",
    requires_input=["smiles"],
    produces="artifact",
    func=_molecular_identity,
))


# ── Tool 2: Assay Curation ─────────────────────────────────────
def _assay_curation(csv_text: str = "", **kwargs) -> dict[str, Any]:
    """Profile and validate an assay CSV table."""
    result: dict[str, Any] = {
        "tool": "assay_curation",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": [],
    }

    if not csv_text:
        result["summary"] = "No CSV data provided. Upload a CSV for assay curation."
        return result

    try:
        df = pd.read_csv(StringIO(csv_text))
        lower = {c.lower(): c for c in df.columns}

        relation = next((lower[k] for k in lower if k in {'relation', 'operator', 'qualifier'}), None)
        unit = next((lower[k] for k in lower if 'unit' in k), None)

        if relation and not df[relation].isin(['<', '>', '=', '>=', '<=']).all():
            result["warnings"].append("Relation column contains unrecognized qualifiers; preserve raw values.")

        if not unit:
            result["warnings"].append("No explicit unit column detected; do not convert concentrations automatically.")

        result["data"] = {
            "rows": len(df),
            "columns": list(df.columns),
            "duplicates": int(df.duplicated().sum()),
            "relation_column": relation,
            "unit_column": unit,
            "missing_values": df.isna().sum().to_dict(),
        }
        result["status"] = "success"
        result["summary"] = f"Assay table profiled: {len(df)} rows, {len(df.columns)} columns"
    except Exception as e:
        result["status"] = "error"
        result["summary"] = str(e)

    return result


register_tool(ToolMeta(
    name="assay_curation",
    description="Profile and QC an assay CSV table — detect relation/unit columns, duplicates, missing values. Requires CSV text input.",
    domain="chemistry",
    requires_input=["csv_text"],
    produces="artifact",
    func=_assay_curation,
))


# ── Tool 3: Chemical Similarity Search ─────────────────────────
def _chemical_similarity(smiles_a: str = "", smiles_b: str = "", **kwargs) -> dict[str, Any]:
    """Compute Tanimoto similarity between two SMILES."""
    result: dict[str, Any] = {
        "tool": "chemical_similarity",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": [],
    }

    if not smiles_a or not smiles_b:
        # Try to extract two SMILES from the question
        question = kwargs.get("question", "")
        # Only match SMILES-like strings (contain typical SMILES characters)
        smiles_list = re.findall(r'[A-Za-z0-9@\[\]\(\)\\\/=#+\-]{5,}', question)
        # Filter out common English words and non-chemical tokens
        common_words = {'smiles', 'compare', 'similarity', 'tanimoto', 'molecule', 'compound',
                        'identify', 'analyze', 'structure', 'chemical', 'between', 'distance'}
        smiles_list = [s for s in smiles_list if s.lower() not in common_words]
        # Keep only strings that look like SMILES (contain C, O, N, etc. with special chars)
        smiles_like = [s for s in smiles_list if re.search(r'[=#@\[\]\(\)\\\/]|[A-Z][a-z]', s)]
        if len(smiles_like) >= 2:
            smiles_a, smiles_b = smiles_like[0], smiles_like[1]
        else:
            result["summary"] = "Provide two valid SMILES strings to compare (e.g., CCO and CC=O)."
            return result

    try:
        from rdkit import Chem
        from rdkit.Chem import DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit import RDLogger
        RDLogger.logger().setLevel(RDLogger.ERROR)  # Suppress parse warnings

        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            result["status"] = "error"
            result["summary"] = "One or both SMILES are invalid."
            return result

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp_a = gen.GetFingerprint(mol_a)
        fp_b = gen.GetFingerprint(mol_b)
        tanimoto = DataStructs.TanimotoSimilarity(fp_a, fp_b)

        result["data"] = {
            "smiles_a": Chem.MolToSmiles(mol_a, canonical=True),
            "smiles_b": Chem.MolToSmiles(mol_b, canonical=True),
            "tanimoto_similarity": round(tanimoto, 4),
        }
        result["status"] = "success"
        result["summary"] = f"Tanimoto similarity (Morgan r=2): {tanimoto:.4f}"
    except ImportError:
        result["status"] = "error"
        result["summary"] = "RDKit not installed."
    except Exception as e:
        result["status"] = "error"
        result["summary"] = str(e)

    return result


register_tool(ToolMeta(
    name="chemical_similarity",
    description="Compute Tanimoto similarity between two chemical structures using Morgan fingerprints (r=2, 2048 bits). Provide two SMILES.",
    domain="chemistry",
    requires_input=["smiles_a", "smiles_b"],
    produces="artifact",
    func=_chemical_similarity,
))


# ── Tool: Chemistry Knowledge (LLM-powered, ALWAYS fires) ──────
register_tool(make_knowledge_tool(
    domain="chemistry",
    tool_name="chemistry_knowledge",
    description="Answer ANY chemistry question: reaction mechanisms, synthesis strategy, spectroscopy, computational chemistry, green chemistry, pKa prediction, catalysis — open-ended. Always runs.",
    domain_label="Chemistry",
    system_prompt=CHEMISTRY_KNOWLEDGE_PROMPT,
))


# ── Tool: Retrosynthesis Planner (LLM-powered) ─────────────────
def _retrosynthesis_planner(question: str = "", **kwargs) -> dict[str, Any]:
    """LLM-powered retrosynthetic analysis."""
    from langgraph_app.graph import _get_llm
    from langchain_core.messages import HumanMessage

    result: dict[str, Any] = {
        "tool": "retrosynthesis_planner",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": ["Retrosynthetic suggestions are computational. Experimental validation required."],
    }

    if not question or not any(w in question.lower() for w in
        ['synthes', 'retrosynth', 'pathway', 'route', 'make ', 'prepare ',
         'starting material', 'precursor', 'disconnection']):
        result["summary"] = "No retrosynthesis query detected. Ask about synthetic routes, disconnections, or starting materials."
        return result

    try:
        llm = _get_llm()
        prompt = f"""You are an expert synthetic organic chemist. Propose a retrosynthetic analysis.

QUERY: {question}

Provide:
1. TARGET ANALYSIS: key functional groups, stereochemistry, ring systems
2. KEY DISCONNECTIONS: 2-4 strategic bond disconnections
3. SYNTHETIC ROUTE: step-by-step forward synthesis from commercial starting materials
4. CRITICAL STEPS: reactions requiring optimization
5. ALTERNATIVES: 1-2 alternative strategies to consider

Be specific with named reactions, reagents, and conditions."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, 'content') else str(resp)

        result["data"] = {"analysis": text, "method": "LLM retrosynthetic analysis"}
        result["status"] = "success"
        result["summary"] = text[:300]
    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Retrosynthesis analysis failed: {e}"

    return result


register_tool(ToolMeta(
    name="retrosynthesis_planner",
    description="LLM-powered retrosynthetic analysis. Propose disconnections, synthetic routes, and alternative strategies. Use for: 'How would you synthesize...', 'Retrosynthesis of...', 'Synthetic route to...'",
    domain="chemistry",
    requires_input=["question"],
    produces="artifact",
    requires_llm=True,
    func=_retrosynthesis_planner,
))


# ── Tool: Spectroscopy Interpreter (LLM-powered) ───────────────
def _spectroscopy_interpreter(question: str = "", **kwargs) -> dict[str, Any]:
    """LLM interprets NMR, IR, MS, UV-Vis data."""
    from langgraph_app.graph import _get_llm
    from langchain_core.messages import HumanMessage

    result: dict[str, Any] = {
        "tool": "spectroscopy_interpreter",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": ["LLM spectral interpretation — verify with experimental data and reference spectra."],
    }

    spec_keywords = ['nmr', 'ir ', 'infrared', 'mass spec', 'lc-ms', 'gc-ms', 'uv-vis',
                     'spectrum', 'spectra', 'chemical shift', 'coupling', 'peak ',
                     'fragment', 'm/z', 'wavenumber', 'absorbance']
    if not question or not any(w in question.lower() for w in spec_keywords):
        result["summary"] = "No spectroscopy query detected. Ask about NMR, IR, MS, UV-Vis interpretation."
        return result

    try:
        llm = _get_llm()
        prompt = f"""You are an expert in spectroscopic analysis (NMR, IR, MS, UV-Vis, LC-MS, GC-MS).

QUERY: {question}

Analyze the spectral data if provided. If peaks/shifts are given:
1. Identify key signals (chemical shifts, coupling constants, IR bands, mass fragments)
2. Propose functional groups present
3. Suggest possible structures consistent with the data
4. Note ambiguities — what additional experiments would resolve them?

If the query is theoretical (e.g., "How to interpret..."): explain the approach step-by-step."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, 'content') else str(resp)

        result["data"] = {"interpretation": text, "method": "LLM spectral interpretation"}
        result["status"] = "success"
        result["summary"] = text[:300]
    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Spectroscopy interpretation failed: {e}"

    return result


register_tool(ToolMeta(
    name="spectroscopy_interpreter",
    description="LLM-powered spectroscopy interpreter. Analyze NMR, IR, MS, UV-Vis, LC-MS, GC-MS data. Provide peaks/shifts for structural elucidation.",
    domain="chemistry",
    requires_input=["question"],
    produces="artifact",
    requires_llm=True,
    func=_spectroscopy_interpreter,
))


# ── Tool: Reaction Predictor (LLM-powered) ─────────────────────
def _reaction_predictor(question: str = "", **kwargs) -> dict[str, Any]:
    """LLM predicts reaction products, yields, and side products."""
    from langgraph_app.graph import _get_llm
    from langchain_core.messages import HumanMessage

    result: dict[str, Any] = {
        "tool": "reaction_predictor",
        "status": "abstain",
        "summary": "",
        "data": {},
        "warnings": ["LLM reaction predictions — validate experimentally."],
    }

    rxn_keywords = ['reaction', 'product', 'yield', 'mechanism', 'side product',
                    'reagent', 'catalyst', 'condition', 'predict', 'selectiv',
                    'coupling', 'oxidation', 'reduction', 'substitution', 'elimination',
                    'addition', 'rearrangement', 'protect', 'deprotect']
    if not question or not any(w in question.lower() for w in rxn_keywords):
        result["summary"] = "No reaction prediction query detected. Ask about reaction products, yields, mechanisms, or selectivity."
        return result

    try:
        llm = _get_llm()
        prompt = f"""You are an expert synthetic organic chemist. Predict reaction outcomes.

QUERY: {question}

Provide:
1. MECHANISM: key steps of the reaction mechanism
2. MAJOR PRODUCT(S): structures (as SMILES), expected ratio if multiple
3. SIDE PRODUCTS: likely impurities and their origin
4. YIELD EXPECTATION: typical yield range with explanation
5. SELECTIVITY: regio-, stereo-, chemo-selectivity considerations
6. OPTIMIZATION SUGGESTIONS: how to improve yield/selectivity

If the query is about a failed reaction: suggest troubleshooting steps."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, 'content') else str(resp)

        result["data"] = {"prediction": text, "method": "LLM reaction prediction"}
        result["status"] = "success"
        result["summary"] = text[:300]
    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Reaction prediction failed: {e}"

    return result


register_tool(ToolMeta(
    name="reaction_predictor",
    description="LLM-powered reaction predictor. Predict products, yields, selectivity, side products. Troubleshoot failed reactions.",
    domain="chemistry",
    requires_input=["question"],
    produces="artifact",
    requires_llm=True,
    func=_reaction_predictor,
))
