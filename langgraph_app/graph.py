"""
LangGraph LifeScienceBench — Supervisor-Agent State Graph.

Architecture:
    discover → router ──user-selected?──→ execute_teams ──blocked?──→ report
                    │                          │
                    │ auto                     │ success
                    ↓                          ↓
              supervisor_plan          supervisor_review
                    │                          │
                    └────→ execute_teams ←─────┘
                                                    ↓
                                                  discuss → report → END

Key design:
- LLM-as-a-judge ONLY for domain routing — zero keywords, zero regex
- Teams: run tools directly (self-detecting via regex/parsing), then LLM synthesizes
- No __pending__ stash, no sequential dispatch — all teams run at once
- Supervisor reviews ALL results together, then board reviews
- User-explicit mode: skip supervisor, go straight to selected teams
"""

import json
import re as _re
from typing import Any

# ── Import-time registration ──────────────────────────────
import langgraph_app.tools.chemistry_tools   # noqa: F401
import langgraph_app.tools.biology_tools     # noqa: F401
import langgraph_app.tools.medical_tools     # noqa: F401
import langgraph_app.tools.rag_tools         # noqa: F401
import langgraph_app.tools.knowledge_tools   # noqa: F401 — LLM-powered knowledge tools
import langgraph_app.agents.chemistry        # noqa: F401
import langgraph_app.agents.biology          # noqa: F401
import langgraph_app.agents.medical          # noqa: F401
import langgraph_app.agents.literature_review  # noqa: F401
import langgraph_app.agents.discussion       # noqa: F401
import langgraph_app.agents.drug_discovery   # noqa: F401
import langgraph_app.agents.bioinformatics   # noqa: F401
import langgraph_app.agents.deep_research    # noqa: F401
import langgraph_app.agents.scientific_writer  # noqa: F401
import langgraph_app.agents.statistics_advisor  # noqa: F401
import langgraph_app.agents.research_planner  # noqa: F401
import langgraph_app.agents.literature_agents  # noqa: F401
import langgraph_app.agents.review_agents      # noqa: F401
import langgraph_app.agents.experiment_agents  # noqa: F401
import langgraph_app.agents.clinical_agents    # noqa: F401
import langgraph_app.agents.ip_agents          # noqa: F401
import langgraph_app.agents.communication_agents  # noqa: F401

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_app.state import AgentState
from langgraph_app.agents.registry import (
    get_agent,
    get_agents_by_domain,
    list_agent_names,
)
from langgraph_app.tools.registry import (
    get_tool,
    get_tools_by_domain,
    list_tool_names,
)
from langgraph_app.config import (
    LLM_PROVIDER, MODEL_ID,
    DEEPSEEK_API_KEY, DEEPSEEK_PRO_MODEL, DEEPSEEK_BASE_URL,
)


# ═══════════════════════════════════════════════════════════════
# LLM Factory
# ═══════════════════════════════════════════════════════════════

_llm_cache: dict[str, Any] = {}


def _get_llm():
    """Get or create an LLM instance. Cached across runs."""
    cache_key = f"{LLM_PROVIDER}:{DEEPSEEK_PRO_MODEL if LLM_PROVIDER == 'deepseek' else MODEL_ID}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    if LLM_PROVIDER == "deepseek":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=DEEPSEEK_PRO_MODEL, api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL, temperature=0,
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        from langgraph_app.config import OPENAI_API_KEY, OPENAI_BASE_URL
        kwargs = {"model": MODEL_ID, "temperature": 0, "seed": 42}
        if OPENAI_API_KEY:
            kwargs["api_key"] = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        llm = ChatOpenAI(**kwargs)
    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        from langgraph_app.config import GROQ_API_KEY
        llm = ChatGroq(model=MODEL_ID, api_key=GROQ_API_KEY, temperature=0)
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from langgraph_app.config import ANTHROPIC_API_KEY
        llm = ChatAnthropic(model=MODEL_ID, api_key=ANTHROPIC_API_KEY, temperature=0)
    elif LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        from langgraph_app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")
    _llm_cache[cache_key] = llm
    return llm


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _get_team_tools(team_name: str) -> list[str]:
    meta = get_agent(team_name)
    return meta.tool_names if meta else []


def _extract_json(text: str) -> str | None:
    m = _re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        return m.group(1).strip()
    m = _re.search(r'\{[\s\S]*\}', text)
    return m.group(0).strip() if m else None


TEAM_DESCRIPTIONS = """
⚗ chemistry — Molecular cheminformatics. Compute properties (MW, LogP, TPSA, HBD/HBA, rotatable bonds, InChIKey) from SMILES. Compare compounds by Tanimoto similarity. Profile assay CSV data. **Also answers ANY chemistry question** from general knowledge: reaction mechanisms, retrosynthesis, spectroscopy interpretation, pKa prediction, solvent selection, green chemistry. Tools: molecular_identity, chemical_similarity, assay_curation, chemistry_knowledge, retrosynthesis_planner, spectroscopy_interpreter, reaction_predictor.

🧬 biology — Protein/peptide sequence QC (composition, length, aromatic/charged ratios). Parse multi-record FASTA files. Profile omics data. **Also answers ANY biology question**: molecular/cell biology, pathways, CRISPR, genetics, immunology, systems biology. Tools: protein_sequence_qc, fasta_parser, omics_qc, biology_knowledge.

🏥 medical — Medical research support. SAFETY-GATED: clinical actions (diagnose, treat, prescribe) are BLOCKED. FHIR resource validation. PICO element extraction. **Also answers ANY medical research question**: disease mechanisms, clinical evidence, pharmacology, epidemiology, guidelines. Tools: medical_safety_gate, fhir_qc, pico_extraction, medical_knowledge.

📚 literature_review — Scientific literature review. Hybrid retrieval from local corpus with citation tracking. Evidence synthesis and gap analysis. **Also answers ANY literature question**: landmark papers, search strategies, critical appraisal, evidence grading. Tools: hybrid_retrieval, corpus_ingest, literature_knowledge.

💊 drug_discovery — Drug discovery expert. Target ID/validation, hit finding, lead optimization, ADMET, medicinal chemistry, computational drug design, clinical development strategy. **Answers ANY drug discovery question** from general knowledge. Tools: drug_discovery_knowledge.

🖥 bioinformatics — Bioinformatics expert. NGS pipelines, variant calling, pathway enrichment, genome assembly, single-cell analysis, phylogenetics, tool/pipeline recommendations. **Answers ANY bioinformatics question**. Tools: bioinformatics_knowledge.

🔬 deep_research — Systematic evidence synthesis. Search strategies, evidence grading (GRADE/OCEBM), meta-analysis methodology, contradiction identification, gap analysis. **Thorough, evidence-graded answers**. Tools: deep_research_knowledge.

✍ scientific_writer — Scientific writing assistant. Draft abstracts, manuscripts, grant proposals, review articles, figure legends. **AI-generated draft text — always verify and edit**. Tools: scientific_writer_knowledge.

📊 statistics_advisor — Biostatistics advisor. Study design, test selection, power analysis, regression, survival analysis, Bayesian methods, meta-analysis, ML evaluation. **Consult a biostatistician for final decisions**. Tools: statistics_knowledge.

🗺 research_planner — Research strategist. PhD roadmaps, grant strategy, experiment prioritization, publication strategy, career development. Like a senior PI advising. Tools: research_planner_knowledge.

── 📖 LITERATURE & EVIDENCE TEAM ──

📄 paper_summarizer — Summarize one or many papers. Extract research question, methods, key results, conclusions, significance. Structured comparison for multiple papers. Tools: paper_summarizer_knowledge.

🔗 evidence_synthesizer — Build consensus from multiple studies. Weigh by quality and sample size, identify where evidence preponderance lies, grade overall strength. Tools: evidence_synthesizer_knowledge.

⚡ contradiction_finder — Find studies with conflicting results. Analyze why they disagree (methods, populations, confounders) and assess which evidence is stronger. Tools: contradiction_finder_knowledge.

🔍 citation_explorer — Find landmark papers and recent breakthroughs. Trace citation networks, identify seminal works and influential recent publications. Tools: citation_explorer_knowledge.

📋 journal_club — Critically analyze a publication as if presenting at journal club. Evaluate hypothesis, methods, results, statistics, conclusions, strengths, and weaknesses. Tools: journal_club_knowledge.

🕳️ research_gap_finder — Identify unexplored opportunities and unanswered questions. Propose specific, testable next studies and high-impact research directions. Tools: research_gap_finder_knowledge.

── 🔬 PEER REVIEW & GRANTS TEAM ──

🔬 peer_reviewer — Review manuscripts like a Nature/Science reviewer. Assess novelty, rigor, reproducibility, statistics, and significance. Constructive critique with actionable suggestions. Tools: peer_reviewer_knowledge.

💰 grant_reviewer — Evaluate funding proposals (NIH R01, ERC, Wellcome). Score significance, innovation, approach, investigators, environment. Identify strengths and weaknesses. Tools: grant_reviewer_knowledge.

── 🧪 EXPERIMENT & PROTOCOL TEAM ──

💡 hypothesis_generator — Generate novel, testable scientific hypotheses. Propose mechanisms, rank by plausibility, suggest discriminating experiments to distinguish between hypotheses. Tools: hypothesis_generator_knowledge.

🧪 experiment_designer — Design experiments to validate findings. Specify controls, sample sizes, methods, expected outcomes, and potential pitfalls. Covers in vitro, in vivo, and clinical studies. Tools: experiment_designer_knowledge.

🔧 protocol_optimizer — Improve experimental protocols (Western blot, PCR, ELISA, cell culture, chromatography, synthesis). Suggest reagent concentrations, incubation times, temperature optimization, troubleshooting. Tools: protocol_optimizer_knowledge.

🛠️ troubleshooting — Diagnose failed experiments. Identify likely causes (reagents, contamination, temperature, pH, instrument), suggest specific fixes. PCR, cloning, Western blot, cell culture, synthesis, crystallization. Tools: troubleshooting_knowledge.

── 🏥 CLINICAL & REGULATORY TEAM ──

📊 clinical_trial_analyst — Analyze and compare clinical trials. Phase I-III design, endpoints, statistics, populations, results interpretation. Cross-trial comparisons. Tools: clinical_trial_analyst_knowledge.

📋 regulatory_advisor — FDA/EMA/ICH regulatory guidance. IND/NDA requirements, expedited pathways, biomarker qualification, trial design for registration. Informational only — consult official guidance. Tools: regulatory_advisor_knowledge.

🛡️ safety_reviewer — Toxicology assessment. hERG, CYP inhibition, genotoxicity, DILI, structural alerts. AI screening — not GLP safety data. Recommend follow-up assays. Tools: safety_reviewer_knowledge.

── 💼 IP & BUSINESS TEAM ──

📜 patent_search — Patent landscape analysis. Composition-of-matter, method, formulation patents. Assignees, expiration dates, white space. AI-estimated — not FTO or legal advice. Tools: patent_search_knowledge.

🏢 competitive_intelligence — Pharma pipeline comparison. Competitor analysis, deal flow, patent cliffs, therapeutic area trends. AI-estimated from public data. Tools: competitive_intelligence_knowledge.

── 🎨 COMMUNICATION & EDUCATION TEAM ──

🎨 figure_generator — Design publication figures and graphical abstracts. Provides detailed concepts and descriptions — rendering requires BioRender, GraphPad, matplotlib, Illustrator. Tools: figure_generator_knowledge.

🎤 presentation_coach — Prepare conference talks. Structure 15/30 min presentations, design slide flow, anticipate questions, craft narrative arc. Tools: presentation_coach_knowledge.

📖 teaching_assistant — Explain scientific concepts at undergraduate, graduate, or public level. Analogies, progressive complexity, connections to real-world applications. Tools: teaching_assistant_knowledge.

📓 lab_notebook — Organize experiments into structured entries. ELN best practices, timestamps, reagent tracking, cross-references, action items. Tools: lab_notebook_knowledge.
"""


# ═══════════════════════════════════════════════════════════════
# Node: discover
# ═══════════════════════════════════════════════════════════════

def _discover_node(state: AgentState) -> dict[str, Any]:
    return {
        "available_agents": list_agent_names(),
        "available_tools": list_tool_names(),
        "supervisor_iteration": 0,
    }


# ═══════════════════════════════════════════════════════════════
# Router: user-selected → execute_teams  |  auto → supervisor_plan
# ═══════════════════════════════════════════════════════════════

def _entry_router(state: AgentState) -> Command:
    selected = state.get("selected_agents", [])
    if selected and len(selected) > 0:
        return Command(
            goto="execute_teams",
            update={
                "supervisor_thought": f"User-selected teams: {', '.join(selected)}",
                "next_action": "continue",
            }
        )
    return Command(goto="supervisor_plan")


# ═══════════════════════════════════════════════════════════════
# Node: supervisor_plan — LLM-as-a-judge ONLY
# ═══════════════════════════════════════════════════════════════

SUPERVISOR_PLAN_PROMPT = f"""You are the **Supervisor** of a multi-agent scientific research system.
Your ONLY job: decide which specialized teams to dispatch.

## Available Teams
{TEAM_DESCRIPTIONS}

## Output — JSON ONLY, no markdown, no explanation:
{{"thought": "why these teams?", "teams": ["team1", "team2"]}}

Pick 1-3 teams. If unsure: use ["literature_review"]. Output ONLY the JSON object."""


def _supervisor_plan_node(state: AgentState) -> Command:
    question = state.get("question", "")
    teams = []
    thought = "LLM unavailable — defaulting to literature_review."

    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(
            content=f"{SUPERVISOR_PLAN_PROMPT}\n\nQuestion: {question}"
        )])
        text = response.content if hasattr(response, 'content') else str(response)
        js = _extract_json(text)
        if js:
            plan = json.loads(js)
            teams = plan.get("teams", [])
            thought = plan.get("thought", thought)
    except Exception as e:
        thought = f"LLM routing failed ({e})."

    valid = set(list_agent_names())
    selected = [t for t in teams if t in valid]
    if not selected:
        selected = ["literature_review"]

    return Command(
        goto="execute_teams",
        update={
            "selected_agents": selected,
            "supervisor_thought": thought,
            "next_action": "continue",
            "supervisor_iteration": 1,
        }
    )


# ═══════════════════════════════════════════════════════════════
# Node: execute_teams — all selected teams, tools + LLM synthesis
# ═══════════════════════════════════════════════════════════════

def _execute_teams_node(state: AgentState) -> dict[str, Any]:
    selected = state.get("selected_agents", [])
    question = state.get("question", "")
    agent_outputs = {}
    all_evidence = []

    for team_name in selected:
        result = _run_single_team(team_name, question)
        agent_outputs[team_name] = result
        retrieval = result.get("tool_results", {}).get("hybrid_retrieval", {})
        chunks = retrieval.get("data", {}).get("chunks", [])
        all_evidence.extend(chunks)

    blocked = any(v.get("status") == "blocked" for v in agent_outputs.values())
    return {
        "agent_outputs": agent_outputs,
        "evidence": all_evidence,
        "next_action": "stop" if blocked else "continue",
        "error": "Medical safety gate blocked execution." if blocked else None,
    }


def _run_single_team(team_name: str, question: str) -> dict[str, Any]:
    """1) Run tools directly  2) LLM synthesizes  3) Build summary."""
    output: dict[str, Any] = {
        "agent": team_name, "status": "success", "summary": "",
        "tool_results": {}, "llm_reasoning": "", "warnings": [],
    }

    # Step 1: Run tools
    for tname in _get_team_tools(team_name):
        tmeta = get_tool(tname)
        if not tmeta or not tmeta.func:
            continue
        try:
            output["tool_results"][tname] = tmeta.func(question=question)
        except Exception as e:
            output["tool_results"][tname] = {"status": "error", "summary": str(e)}

    # Step 2: LLM synthesis (ALWAYS run — LLM can answer from general knowledge)
    has_tool_results = any(
        v.get("status") not in ("abstain", "error", None)
        for v in output["tool_results"].values()
    )
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage

        # Build context from tool results
        tool_ctx_parts = []
        for k, v in output["tool_results"].items():
            tool_ctx_parts.append(f"[{k}] ({v.get('status', '?')}): {json.dumps(v, default=str)[:400]}")
        tool_ctx = "\n".join(tool_ctx_parts)

        if has_tool_results:
            prompt = f"""You are a {team_name} scientific assistant.

QUESTION: {question}

TOOL RESULTS:
{tool_ctx}

Synthesize these results into a clear, direct answer (2-4 sentences). Cite specific values. Do NOT invent data."""
        else:
            prompt = f"""You are a {team_name} scientific assistant.

QUESTION: {question}

TOOL RESULTS (all abstained — no tools matched the query):
{tool_ctx}

TASK: Answer the question from your general scientific knowledge (2-4 sentences).
If you know specific values (e.g., molecular weight of aspirin = 180.16 g/mol), cite them.
Note that the computational tools could not be applied, so your answer is based on general knowledge.
If you need SMILES to compute, suggest the SMILES string."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        output["llm_reasoning"] = resp.content if hasattr(resp, 'content') else str(resp)
    except Exception as e:
        output["llm_reasoning"] = f"LLM synthesis failed ({e})."

    # Step 3: Summary
    successes = [k for k, v in output["tool_results"].items() if v.get("status") == "success"]
    errors = [k for k, v in output["tool_results"].items() if v.get("status") == "error"]
    blocked = [k for k, v in output["tool_results"].items() if v.get("status") == "blocked"]
    abstains = [k for k, v in output["tool_results"].items() if v.get("status") == "abstain"]

    if blocked:
        output["status"] = "blocked"
        output["summary"] = output["tool_results"][blocked[0]].get("summary", "Blocked.")
        output["warnings"] = output["tool_results"][blocked[0]].get("warnings", [])
    elif successes:
        output["summary"] = f"{team_name}: {', '.join(successes)} succeeded"
        if output["llm_reasoning"]:
            output["summary"] += " — LLM synthesis available"
    elif errors:
        output["status"] = "error"
        output["summary"] = f"Tools failed: {', '.join(errors)}"
    elif abstains:
        output["status"] = "success"  # LLM synthesis provides the answer
        output["summary"] = f"{team_name}: tools did not match, LLM answered from general knowledge"
        if output["llm_reasoning"]:
            output["summary"] += " — synthesis available"
    else:
        output["status"] = "abstain"
        output["summary"] = "No tools available for this team."

    warnings_map = {
        "chemistry": "Computational estimates only. No activity/toxicity/clinical inference.",
        "biology": "Sequence composition only. No functional/structural/clinical inference.",
        "medical": "Research support only. No clinical decisions.",
        "literature_review": "Local corpus only — not comprehensive.",
    }
    if team_name in warnings_map:
        output["warnings"].append(warnings_map[team_name])
    return output


# ═══════════════════════════════════════════════════════════════
# Router: blocked → report  |  success → supervisor_review
# ═══════════════════════════════════════════════════════════════

def _after_execute_router(state: AgentState) -> Command:
    if state.get("next_action") == "stop":
        return Command(goto="report")
    return Command(goto="supervisor_review")


# ═══════════════════════════════════════════════════════════════
# Node: supervisor_review — reviews ALL team results at once
# ═══════════════════════════════════════════════════════════════

def _supervisor_review_node(state: AgentState) -> dict[str, Any]:
    question = state.get("question", "")
    agent_outputs = state.get("agent_outputs", {})

    parts = []
    for tname, out in agent_outputs.items():
        parts.append(f"### {tname} ({out.get('status', '?')})")
        parts.append(f"Summary: {out.get('summary', '')}")
        if out.get("llm_reasoning"):
            parts.append(f"Analysis: {out['llm_reasoning'][:600]}")
        for tn, tr in out.get("tool_results", {}).items():
            parts.append(f"  [{tn}]: {tr.get('summary', '')[:200]}")
        parts.append("")
    team_results = "\n".join(parts)

    synthesis = ""
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage
        prompt = f"""Review team outputs for this research question.

QUESTION: {question}

TEAM RESULTS:
{team_results}

Output JSON ONLY: {{"assessment": "1-2 sentence assessment", "gaps": ["gap1"], "recommendation": "next step"}}"""
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, 'content') else str(resp)
        js = _extract_json(text)
        if js:
            review = json.loads(js)
            synthesis = review.get("assessment", "")
            gaps = review.get("gaps", [])
            rec = review.get("recommendation", "")
            if gaps:
                synthesis += "\n\n**Gaps:** " + "; ".join(gaps)
            synthesis += f"\n\n**Recommendation:** {rec}"
        else:
            synthesis = text[:800]
    except Exception:
        synthesis = "LLM review unavailable. Results as-is."

    return {"synthesis": synthesis, "next_action": "continue"}


# ═══════════════════════════════════════════════════════════════
# Node: discuss — 3-reviewer board
# ═══════════════════════════════════════════════════════════════

def _discuss_node(state: AgentState) -> dict[str, Any]:
    question = state.get("question", "")
    agent_outputs = state.get("agent_outputs", {})
    synthesis = state.get("synthesis", "")
    evidence = state.get("evidence", [])
    blocked = any(v.get("status") == "blocked" for v in agent_outputs.values())

    if blocked:
        return {"board_review": "# Review Board\n\nExecution blocked by safety gate.", "next_action": "continue"}

    ctx_parts = [f"RESEARCH QUESTION: {question}\n"]
    if synthesis:
        ctx_parts.append(f"SUPERVISOR SYNTHESIS: {synthesis}\n")
    if evidence:
        ctx_parts.append("## Evidence")
        for e in evidence[:5]:
            ctx_parts.append(f"- [{e.get('citation', '?')}] {e.get('excerpt', '')[:200]}")
    for aname, aout in agent_outputs.items():
        if aname == "discussion":
            continue
        ctx_parts.append(f"\n## {aname} Team ({aout.get('status', '?')})")
        ctx_parts.append(f"Summary: {aout.get('summary', '')}")
        if aout.get("llm_reasoning"):
            ctx_parts.append(f"Analysis: {aout['llm_reasoning'][:400]}")
    context = "\n".join(ctx_parts)

    roles = [
        ("domain_scientist", "Domain Scientist", "Scientific plausibility, domain consistency."),
        ("methods_critic", "Methods & Statistics Critic", "Methodological rigor, reproducibility."),
        ("data_quality_reviewer", "Data Quality Reviewer", "Data provenance, quality, completeness."),
    ]
    reviewers = {}
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage, SystemMessage
        for rkey, rlabel, rfocus in roles:
            prompt = f"""You are an independent {rlabel}. Focus: {rfocus}.

{context}

Structure EXACTLY:
FINDINGS: <your assessment>
CONCERNS: <specific concerns>
RECOMMENDATION: <next step>

Be INDEPENDENT. State assumptions. If evidence insufficient, say so — don't invent."""
            resp = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Review.")])
            text = resp.content if hasattr(resp, 'content') else str(resp)
            findings = concerns = recommendation = ""
            for line in text.split('\n'):
                line = line.strip()
                if line.upper().startswith('FINDINGS:'):
                    findings = line.split(':', 1)[1].strip()
                elif line.upper().startswith('CONCERNS:'):
                    concerns = line.split(':', 1)[1].strip()
                elif line.upper().startswith('RECOMMENDATION:'):
                    recommendation = line.split(':', 1)[1].strip()
            reviewers[rkey] = {
                "label": rlabel, "findings": findings or text[:200],
                "concerns": concerns or "Not specified",
                "recommendation": recommendation or "Not specified",
            }
    except Exception as e:
        for rkey, rlabel, rfocus in roles:
            reviewers[rkey] = {
                "label": rlabel, "findings": f"LLM unavailable ({e}).",
                "concerns": "N/A", "recommendation": "Independent expert review.",
            }

    lines = ["# Independent Review Board\n"]
    lines.append(f"**{len(reviewers)} reviewers** assessed the evidence.\n")
    for rkey, rd in reviewers.items():
        lines.append(f"## {rd['label']}")
        lines.append(f"**Findings:** {rd['findings']}")
        lines.append(f"**Concerns:** {rd['concerns']}")
        lines.append(f"**Recommendation:** {rd['recommendation']}\n")
    lines.append("## Next Actions\n- Validate primary sources\n- Review experimental design/metadata\n- Define a confirmatory experiment")

    return {"board_review": "\n".join(lines), "next_action": "continue"}


# ═══════════════════════════════════════════════════════════════
# Node: report
# ═══════════════════════════════════════════════════════════════

def _report_node(state: AgentState) -> dict[str, Any]:
    question = state.get("question", "")
    agent_outputs = state.get("agent_outputs", {})
    synthesis = state.get("synthesis", "")
    board_review = state.get("board_review", "")
    supervisor_thought = state.get("supervisor_thought", "")
    evidence = state.get("evidence", [])

    lines = ["# LifeScienceBench — Research Report\n", f"## Research Question\n{question}\n"]

    if supervisor_thought:
        lines.append("## Supervisor Routing")
        lines.append(f"_{supervisor_thought}_\n")

    if synthesis:
        lines.append("## Supervisor Synthesis")
        lines.append(f"{synthesis}\n")

    if evidence:
        lines.append("## Retrieved Evidence")
        for e in evidence[:8]:
            lines.append(f"- [{e.get('citation', '?')}] {e.get('excerpt', '')[:200]}")
        lines.append("")

    if agent_outputs:
        lines.append("## Team Results\n")
        for tname, out in agent_outputs.items():
            icon = {"success": "✅", "error": "❌", "blocked": "🚫", "abstain": "⚠️"}.get(out.get("status", ""), "•")
            lines.append(f"### {icon} {tname}")
            lines.append(f"**Summary:** {out.get('summary', '')}")
            if out.get("llm_reasoning"):
                lines.append(f"\n**Analysis:** {out['llm_reasoning'][:600]}")
            for tn, tr in out.get("tool_results", {}).items():
                lines.append(f"- **{tn}**: {tr.get('summary', '')[:200]}")
            lines.append("")

    if board_review:
        lines.append(board_review + "\n")

    lines.append("## Warnings & Limitations")
    seen = set()
    for tname, out in agent_outputs.items():
        for w in out.get("warnings", []):
            if w not in seen:
                seen.add(w)
                lines.append(f"- [{tname}] {w}")
    if not seen:
        lines.append("- Research support tool — not clinical or expert authority.")

    model_str = DEEPSEEK_PRO_MODEL if LLM_PROVIDER == "deepseek" else MODEL_ID
    lines.append(f"\n---\n*LifeScienceBench Supervisor-Agent v0.3.0 · {LLM_PROVIDER}/{model_str}*")

    return {"final_report": "\n".join(lines), "next_action": "stop"}


# ═══════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════

_graph_instance = None


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("discover", _discover_node)
    builder.add_node("router", _entry_router)
    builder.add_node("supervisor_plan", _supervisor_plan_node)
    builder.add_node("execute_teams", _execute_teams_node)
    builder.add_node("after_execute_router", _after_execute_router)
    builder.add_node("supervisor_review", _supervisor_review_node)
    builder.add_node("discuss", _discuss_node)
    builder.add_node("report", _report_node)

    builder.set_entry_point("discover")
    builder.add_edge("discover", "router")
    # router → supervisor_plan or execute_teams (via Command.goto)
    # supervisor_plan → execute_teams (via Command.goto)
    # execute_teams → after_execute_router (via regular edge)
    # after_execute_router → supervisor_review or report (via Command.goto)
    builder.add_edge("execute_teams", "after_execute_router")
    builder.add_edge("supervisor_review", "discuss")
    builder.add_edge("discuss", "report")
    builder.add_edge("report", END)

    return builder.compile(checkpointer=MemorySaver())


def get_graph() -> StateGraph:
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance
