"""
LangGraph LifeScienceBench — FastAPI Service
=============================================
Bridges Open WebUI to the LangGraph supervisor-agent graph.
Exposes the same /runs endpoint as the v0.3.2 Bench API so the
existing lifesciencebench_research pipe works without changes.

Start:  python -m langgraph_app.service
"""

"""
LangGraph LifeScienceBench — FastAPI Service
=============================================
Bridges Open WebUI to the LangGraph supervisor-agent tools.
Uses LLM function-calling for fast responses (1-3 calls, not 15+).

Endpoints:
  GET  /health          — health check with agent/tool listing
  POST /runs            — fast research via LLM function calling
  POST /runs/full       — full LangGraph graph (slower, with discussion board)

Start:  python -m langgraph_app.service
"""

import sys
import os
import json
import traceback
from uuid import uuid4

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Trigger tool & agent registration via imports
import langgraph_app.tools.chemistry_tools   # noqa: F401
import langgraph_app.tools.biology_tools     # noqa: F401
import langgraph_app.tools.medical_tools     # noqa: F401
import langgraph_app.tools.rag_tools         # noqa: F401
import langgraph_app.tools.knowledge_tools   # noqa: F401
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
import langgraph_app.agents.literature_agents  # noqa: F401 — 6 literature sub-agents
import langgraph_app.agents.review_agents      # noqa: F401 — peer reviewer, grant reviewer
import langgraph_app.agents.experiment_agents  # noqa: F401 — hypothesis, experiment, protocol, troubleshooting
import langgraph_app.agents.clinical_agents    # noqa: F401 — clinical trial, regulatory, safety
import langgraph_app.agents.ip_agents          # noqa: F401 — patent search, competitive intelligence
import langgraph_app.agents.communication_agents  # noqa: F401 — figures, presentations, teaching, lab notebook

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph_app.agents.registry import list_agent_names, get_agent
from langgraph_app.tools.registry import list_tool_names, get_tool
from langgraph_app.config import (
    LLM_PROVIDER, DEEPSEEK_PRO_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    MODEL_ID, OPENAI_API_KEY, OPENAI_BASE_URL,
)

app = FastAPI(title="LifeScienceBench LangGraph v0.3.0")


class RunRequest(BaseModel):
    question: str
    mode: str = "auto"
    csv_text: str | None = None
    fhir_text: str | None = None
    project_id: str = "default"
    active_agents: list[str] | None = None
    force_multi_agent: bool = False


# ── Agent → Tool mapping for filtering ────────────────────────────
AGENT_TOOL_MAP: dict[str, list[str]] = {
    "chemistry": [
        "molecular_identity", "chemical_similarity", "assay_curation",
        "chemistry_knowledge", "retrosynthesis_planner",
        "spectroscopy_interpreter", "reaction_predictor",
    ],
    "biology": [
        "protein_sequence_qc", "fasta_parser", "omics_qc",
        "biology_knowledge",
    ],
    "medical": [
        "medical_safety_gate", "fhir_qc", "pico_extraction",
        "medical_knowledge",
    ],
    "literature_review": [
        "hybrid_retrieval", "corpus_ingest", "literature_knowledge",
    ],
    "drug_discovery": ["drug_discovery_knowledge"],
    "bioinformatics": ["bioinformatics_knowledge"],
    "deep_research": ["deep_research_knowledge"],
    "scientific_writer": ["scientific_writer_knowledge"],
    "statistics_advisor": ["statistics_knowledge"],
    "research_planner": ["research_planner_knowledge"],
    # ── Literature & Evidence Team (6) ──
    "paper_summarizer": ["paper_summarizer_knowledge"],
    "evidence_synthesizer": ["evidence_synthesizer_knowledge"],
    "contradiction_finder": ["contradiction_finder_knowledge"],
    "citation_explorer": ["citation_explorer_knowledge"],
    "journal_club": ["journal_club_knowledge"],
    "research_gap_finder": ["research_gap_finder_knowledge"],
    # ── Peer Review & Grants Team (2) ──
    "peer_reviewer": ["peer_reviewer_knowledge"],
    "grant_reviewer": ["grant_reviewer_knowledge"],
    # ── Experiment & Protocol Team (4) ──
    "hypothesis_generator": ["hypothesis_generator_knowledge"],
    "experiment_designer": ["experiment_designer_knowledge"],
    "protocol_optimizer": ["protocol_optimizer_knowledge"],
    "troubleshooting": ["troubleshooting_knowledge"],
    # ── Clinical & Regulatory Team (3) ──
    "clinical_trial_analyst": ["clinical_trial_analyst_knowledge"],
    "regulatory_advisor": ["regulatory_advisor_knowledge"],
    "safety_reviewer": ["safety_reviewer_knowledge"],
    # ── IP & Business Team (2) ──
    "patent_search": ["patent_search_knowledge"],
    "competitive_intelligence": ["competitive_intelligence_knowledge"],
    # ── Communication & Education Team (4) ──
    "figure_generator": ["figure_generator_knowledge"],
    "presentation_coach": ["presentation_coach_knowledge"],
    "teaching_assistant": ["teaching_assistant_knowledge"],
    "lab_notebook": ["lab_notebook_knowledge"],
}

# ── Tool definitions for LLM function calling ─────────────────────
def _build_tool_definitions(active_agents: list[str] | None = None):
    """
    Build OpenAI-format tool definitions from ALL registered tools.
    This includes both data-processing tools (molecular_identity, fhir_qc…)
    and LLM-powered knowledge tools (chemistry_knowledge, biology_knowledge…)
    so the routing LLM can dispatch to the right domain expert.
    """
    from langgraph_app.tools.registry import list_tool_names, get_tool

    # Determine which tools are allowed based on active agents
    if active_agents:
        allowed_tools: set[str] = set()
        for agent in active_agents:
            tools = AGENT_TOOL_MAP.get(agent, [])
            allowed_tools.update(tools)
    else:
        allowed_tools = None  # All tools

    # Map tool input requirements → OpenAI parameter schema
    def _build_params(tmeta):
        """Build JSON Schema parameters from tool's requires_input list."""
        props = {}
        required = []
        for inp in tmeta.requires_input:
            if inp == "question":
                props["question"] = {
                    "type": "string",
                    "description": "The scientific question to answer"
                }
                required.append("question")
            elif inp in ("smiles",):
                props["smiles"] = {
                    "type": "string",
                    "description": "SMILES string of the molecule"
                }
            elif inp in ("csv_text",):
                props["csv_text"] = {
                    "type": "string",
                    "description": "CSV data as text"
                }
            elif inp in ("text", "filename"):
                props[inp] = {
                    "type": "string",
                    "description": f"Input {inp}"
                }
        return {
            "type": "object",
            "properties": props,
            "required": required,
        }

    defs = []
    for tname in list_tool_names():
        # Filter by active agents if specified
        if allowed_tools is not None and tname not in allowed_tools:
            continue
        tmeta = get_tool(tname)
        if tmeta is None:
            continue
        defs.append({
            "type": "function",
            "function": {
                "name": tname,
                "description": tmeta.description or f"Run {tname}",
                "parameters": _build_params(tmeta),
            }
        })
    return defs


def _get_chat_llm():
    """Get LLM client with base_url for DeepSeek."""
    if LLM_PROVIDER == "deepseek":
        from openai import OpenAI
        return OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL + "/v1",
        )
    elif OPENAI_API_KEY:
        from openai import OpenAI
        return OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL or None,
        )
    else:
        raise RuntimeError("No LLM provider configured.")


def _get_model_name():
    return DEEPSEEK_PRO_MODEL if LLM_PROVIDER == "deepseek" else MODEL_ID


# ── Fast path: LLM function calling (1-3 LLM calls) ───────────────

def _run_fast(question: str, active_agents: list[str] | None = None) -> tuple[str, list[dict]]:
    """
    Fast research using LLM function calling.
    1. LLM decides which tool(s) to call
    2. Execute tool(s)
    3. LLM synthesizes final answer
    Total: 1-3 LLM calls = 5-20 seconds.

    Returns:
        (markdown_answer, trace) where trace is a list of event dicts:
        {"phase": "planning"|"tool_call"|"tool_result"|"synthesis"|"done",
         "message": "...", "detail": "...", "elapsed_ms": ...}
    """
    import time
    t0 = time.time()
    trace = []

    def _elapsed():
        return int((time.time() - t0) * 1000)

    client = _get_chat_llm()
    model = _get_model_name()
    tools = _build_tool_definitions(active_agents)

    print(f"\n🚀 _run_fast() — question: {question[:120]}")
    print(f"🚀 active_agents: {active_agents}")
    print(f"🚀 {len(tools)} tools available")
    trace.append({
        "phase": "planning",
        "message": "🔍 Analyzing question & deciding which tools to call…",
        "detail": f"Model: {model}, {len(tools)} tools available",
        "elapsed_ms": _elapsed(),
    })

    system_prompt = (
        "You are LifeScienceBench, an expert scientific research assistant. "
        "Answer questions thoroughly with specific details, data, and citations. "
        "Use markdown formatting for clarity.\n\n"
        "You have access to a suite of specialized tools. Choose wisely:\n\n"
        "**Knowledge tools** (use for open-ended questions in a specific domain):\n"
        "- `chemistry_knowledge` — reaction mechanisms, retrosynthesis, spectroscopy, pKa, solvents, green chemistry\n"
        "- `biology_knowledge` — molecular/cell biology, CRISPR, genetics, immunology, pathways, systems biology\n"
        "- `medical_knowledge` — disease mechanisms, clinical evidence, pharmacology, epidemiology, guidelines\n"
        "- `literature_knowledge` — landmark papers, search strategies, evidence grading, critical appraisal\n"
        "- `drug_discovery_knowledge` — target ID, lead optimization, ADMET, medicinal chemistry\n"
        "- `bioinformatics_knowledge` — NGS, variant calling, pathway enrichment, single-cell analysis\n"
        "- `deep_research_knowledge` — systematic evidence synthesis, GRADE, meta-analysis\n"
        "- `scientific_writer_knowledge` — abstracts, manuscripts, grant proposals\n"
        "- `statistics_knowledge` — study design, test selection, power analysis, regression\n"
        "- `research_planner_knowledge` — PhD roadmaps, grant strategy, experiment prioritization\n\n"
        "**Data-processing tools** (use ONLY when user provides specific data):\n"
        "- `molecular_identity` — compute MW, LogP, TPSA, etc. from a SMILES string\n"
        "- `chemical_similarity` — Tanimoto similarity between SMILES\n"
        "- `protein_sequence_qc` — amino acid composition stats\n"
        "- `fasta_parser` — parse FASTA files\n"
        "- `hybrid_retrieval` — search the local document corpus for evidence\n"
        "- `assay_curation`, `fhir_qc`, `pico_extraction`, `omics_qc` — CSV/FHIR/clinical data processing\n\n"
        "**Rules:**\n"
        "1. For general scientific questions, call the MOST RELEVANT knowledge tool.\n"
        "2. Only call data-processing tools when specific data (SMILES, sequence, CSV) is provided.\n"
        "3. For cross-domain questions, you may call multiple knowledge tools.\n"
        "4. Do NOT call both a knowledge tool AND hybrid_retrieval for the same question — pick one."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Step 1: LLM decides tool calls
    response1 = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.0,
    )
    msg1 = response1.choices[0].message

    print(f"📡 LLM response — tool_calls: {len(msg1.tool_calls) if msg1.tool_calls else 0}, "
          f"content_len: {len(msg1.content or '')}")
    if msg1.tool_calls:
        for tc in msg1.tool_calls:
            print(f"  🔧 {tc.function.name}({tc.function.arguments[:120]})")

    tool_results = []
    if msg1.tool_calls:
        # Log each tool being called
        for tc in msg1.tool_calls:
            tname = tc.function.name
            try:
                targs = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                targs = {"query": question}
            trace.append({
                "phase": "tool_call",
                "message": f"🛠️ Calling tool: **{tname}**",
                "detail": f"Args: {json.dumps(targs, default=str)[:200]}",
                "elapsed_ms": _elapsed(),
            })

        # Append assistant message with all tool_calls (ONCE, before tool results)
        messages.append({
            "role": "assistant",
            "content": msg1.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in msg1.tool_calls
            ]
        })

        # Execute each tool and append tool result messages
        for tc in msg1.tool_calls:
            tname = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"question": question}

            # Ensure question is always available for tools that need it
            if "question" not in args:
                args["question"] = question

            tmeta = get_tool(tname)
            if tmeta and tmeta.func:
                try:
                    result = tmeta.func(**args)
                except Exception as exc:
                    result = {"status": "error", "summary": str(exc)}
                tool_results.append((tname, result))
            else:
                tool_results.append((tname, {"status": "error", "summary": "Tool not found"}))

            # Summarize tool result for trace
            r = tool_results[-1][1]
            if isinstance(r, dict):
                summary = r.get("summary", r.get("status", json.dumps(r, default=str)[:150]))
            else:
                summary = str(r)[:150]
            print(f"  ✅ {tname} — result: {str(summary)[:120]}")
            trace.append({
                "phase": "tool_result",
                "message": f"✅ **{tname}** completed",
                "detail": str(summary)[:300],
                "elapsed_ms": _elapsed(),
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_results[-1][1], default=str),
            })

        # Step 2: Synthesize final answer from tool results
        trace.append({
            "phase": "synthesis",
            "message": "✍️ Synthesizing final answer from tool results…",
            "detail": f"{len(tool_results)} tool result(s) to incorporate",
            "elapsed_ms": _elapsed(),
        })

        # Build synthesis prompt based on which tools were called
        knowledge_tools_called = [t[0] for t in tool_results if t[0].endswith("_knowledge")]
        if knowledge_tools_called:
            synthesis_instruction = (
                "The tool results above contain expert domain knowledge. "
                "Use the ANSWER and REFERENCES from each tool to compose a thorough, "
                "well-structured response. Expand and elaborate on the key points. "
                "Use markdown with tables and headings where helpful. "
                "Be scientific and precise. Include the REFERENCES cited by the tools."
            )
        else:
            synthesis_instruction = (
                "Synthesize a thorough, well-structured answer using the tool results. "
                "Use markdown. Cite specific computed values. Be scientific and precise."
            )
        messages.append({"role": "user", "content": synthesis_instruction})

        response2 = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        answer = response2.choices[0].message.content
    else:
        # No tool calls — use the direct answer
        trace.append({
            "phase": "synthesis",
            "message": "✍️ Answering directly from scientific knowledge (no data to process)…",
            "detail": "No tools needed for this question",
            "elapsed_ms": _elapsed(),
        })
        answer = msg1.content or "No response generated."

    print(f"✅ Synthesis complete — answer length: {len(answer)} chars, total time: {_elapsed()}ms")
    print(f"{'='*60}\n")

    # Build detailed tool-call footer
    trace.append({
        "phase": "done",
        "message": "✅ Research complete",
        "detail": f"Total time: {_elapsed()}ms, {len(tool_results)} tool(s) used",
        "elapsed_ms": _elapsed(),
    })

    if tool_results:
        lines = ["\n\n---\n## 🔬 Research Trace\n"]
        lines.append(f"**Total time:** {_elapsed()}ms  |  **Model:** {model}  |  **Tools called:** {len(tool_results)}\n")
        for i, (tname, result) in enumerate(tool_results, 1):
            lines.append(f"### Step {i}: `{tname}`")
            if isinstance(result, dict):
                status = result.get("status", "")
                summary = result.get("summary", "")
                data = result.get("data", {})
                if status:
                    lines.append(f"- **Status:** {status}")
                # Knowledge tools: show confidence & references
                confidence = data.get("confidence", "")
                refs = data.get("references", "")
                if confidence:
                    lines.append(f"- **Confidence:** {confidence}")
                if refs:
                    lines.append(f"- **References:** {refs[:300]}")
                if summary:
                    lines.append(f"- **Result:** {summary[:500]}")
                # Show document hits if available (hybrid_retrieval)
                docs = result.get("documents", []) or result.get("results", [])
                # hybrid_retrieval stores chunks in data.chunks
                if not docs and isinstance(data, dict):
                    docs = data.get("chunks", [])
                if docs:
                    lines.append(f"- **Documents found:** {len(docs)}")
                    for d in docs[:5]:
                        if isinstance(d, dict):
                            name = d.get("document", d.get("filename", d.get("source", d.get("title", d.get("citation", "")))))
                            score = d.get("score", "")
                            excerpt = d.get("excerpt", "")
                            if name:
                                line = f"  - `{name}`"
                                if isinstance(score, (int, float)):
                                    line += f" (score: {score:.3f})"
                                lines.append(line)
                                if excerpt:
                                    lines.append(f"    > {excerpt[:200]}")
            else:
                lines.append(f"- **Result:** {str(result)[:300]}")
            lines.append("")
        answer += "\n".join(lines)

    return answer, trace


# ── Multi-agent dispatch ────────────────────────────────────────

# Map agent names to their knowledge tool names
AGENT_TOOL_MAP_RUNTIME = {
    "chemistry": "chemistry_knowledge",
    "biology": "biology_knowledge",
    "medical": "medical_knowledge",
    "literature_review": "literature_knowledge",
    "drug_discovery": "drug_discovery_knowledge",
    "bioinformatics": "bioinformatics_knowledge",
    "deep_research": "deep_research_knowledge",
    "scientific_writer": "scientific_writer_knowledge",
    "statistics_advisor": "statistics_knowledge",
    "research_planner": "research_planner_knowledge",
}


def _run_agents_parallel(question: str, active_agents: list[str], t0: float, trace: list) -> dict:
    """Run all active agents in parallel. Returns {agent_name: result_dict}."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    agent_results = {}

    def run_agent(agent_name: str):
        tool_name = AGENT_TOOL_MAP_RUNTIME.get(agent_name)
        if not tool_name:
            return (agent_name, {"status": "error", "summary": f"No tool for {agent_name}"})
        tmeta = get_tool(tool_name)
        if not tmeta or not tmeta.func:
            return (agent_name, {"status": "error", "summary": f"Tool {tool_name} not found"})
        try:
            result = tmeta.func(question=question)
            return (agent_name, result)
        except Exception as exc:
            return (agent_name, {"status": "error", "summary": str(exc)})

    with ThreadPoolExecutor(max_workers=min(len(active_agents), 5)) as executor:
        futures = {executor.submit(run_agent, a): a for a in active_agents}
        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                name, result = future.result()
                agent_results[name] = result
                elapsed = int((time.time() - t0) * 1000)
                trace.append({
                    "phase": "tool_result",
                    "message": f"✅ **{name}** completed",
                    "detail": str(result.get("summary", ""))[:300] if isinstance(result, dict) else str(result)[:200],
                    "elapsed_ms": elapsed,
                })
            except Exception as exc:
                agent_results[agent_name] = {"status": "error", "summary": str(exc)}

    return agent_results


def _agent_outputs_to_text(agent_results: dict) -> str:
    """Convert agent results dict to a text block for LLM prompts."""
    parts = []
    for agent_name, result in agent_results.items():
        if isinstance(result, dict):
            data = result.get("data", {})
            answer = data.get("answer", "") or result.get("summary", "")
            references = data.get("references", "")
            confidence = data.get("confidence", "")
            if answer:
                header = f"## {agent_name.replace('_', ' ').title()} (confidence: {confidence})"
                parts.append(f"{header}\n\n{answer[:3000]}")
                if references:
                    parts.append(f"\n*References:* {references[:500]}")
    return "\n\n".join(parts) if parts else "No agent outputs available."


def _run_multi_agent(question: str, active_agents: list[str], deep: bool = False) -> tuple[str, list[dict], list[str]]:
    """
    Multi-agent dispatch. If deep=True, runs 3 rounds:
      Round 1 — Parallel agent answers
      Round 2 — Cross-review critique & gap analysis
      Round 3 — Deep synthesis incorporating all rounds
    Otherwise single-round dispatch + synthesis.
    """
    import time
    t0 = time.time()

    def _elapsed():
        return int((time.time() - t0) * 1000)

    trace = []
    total_rounds = 3 if deep else 1

    # ── Round 1: Parallel agent dispatch ───────────────────────
    trace.append({
        "phase": "planning",
        "message": f"🔍 Round 1/{total_rounds}: Dispatching {len(active_agents)} agents…",
        "detail": f"Agents: {', '.join(active_agents)}",
        "elapsed_ms": _elapsed(),
    })

    agent_results = _run_agents_parallel(question, active_agents, t0, trace)
    round1_text = _agent_outputs_to_text(agent_results)

    agent_messages = []
    for agent_name, result in agent_results.items():
        if isinstance(result, dict):
            data = result.get("data", {})
            conf = data.get("confidence", "") or result.get("confidence", "N/A")
            agent_messages.append(f"🤖 **{agent_name}**: {conf} confidence")

    if not deep:
        # ── Single-round synthesis (original behavior) ─────────
        trace.append({
            "phase": "synthesis",
            "message": "✍️ Synthesizing multi-agent results…",
            "detail": f"{len(agent_results)} agent(s) completed",
            "elapsed_ms": _elapsed(),
        })

        client = _get_chat_llm()
        model = _get_model_name()
        synthesis_prompt = (
            "You are LifeScienceBench, an expert scientific research synthesizer. "
            "Below are outputs from multiple domain-expert agents. Synthesize them into "
            "a single, coherent, well-structured answer. Resolve any conflicts, fill gaps, "
            "and present a unified response. Use markdown with tables and headings where helpful.\n\n"
            f"USER QUESTION: {question}\n\n"
            f"AGENT OUTPUTS:\n{round1_text}\n\n"
            "SYNTHESIZE a comprehensive answer that integrates all agent perspectives."
        )
        synth_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.0,
        )
        answer = synth_resp.choices[0].message.content or "No synthesis generated."

        # Footer
        footer = "\n\n---\n## 🔬 Multi-Agent Trace\n"
        footer += f"**{len(agent_results)} agents consulted** in {_elapsed()}ms\n\n"
        for agent_name, result in agent_results.items():
            if isinstance(result, dict):
                data = result.get("data", {})
                conf = data.get("confidence", "") or result.get("confidence", "N/A") if isinstance(result.get("data"), dict) else result.get("confidence", "N/A")
                refs = data.get("references", "") if isinstance(result.get("data"), dict) else ""
                footer += f"- **{agent_name}**: confidence={conf}"
                if refs:
                    footer += f", refs={refs[:200]}"
                footer += "\n"
        answer += footer

        trace.append({
            "phase": "done",
            "message": "✅ Multi-agent research complete",
            "detail": f"Total time: {_elapsed()}ms, {len(agent_results)} agents",
            "elapsed_ms": _elapsed(),
        })
        return answer, trace, agent_messages

    # ── Round 2: Cross-review critique ─────────────────────────
    trace.append({
        "phase": "planning",
        "message": f"🔍 Round 2/3: Cross-review — identifying gaps, conflicts & blind spots…",
        "detail": f"Critiquing {len(agent_results)} agent outputs",
        "elapsed_ms": _elapsed(),
    })

    client = _get_chat_llm()
    model = _get_model_name()

    critique_prompt = (
        "You are a rigorous scientific peer reviewer. Below are outputs from multiple "
        "domain-expert agents answering the same question. Your job is to:\n\n"
        "1. **Identify conflicts** — where do agents disagree? Which interpretation is stronger?\n"
        "2. **Find gaps** — what important aspects did ALL agents miss?\n"
        "3. **Evaluate evidence quality** — which claims are well-supported vs speculative?\n"
        "4. **Suggest integrations** — how can insights from different domains be combined?\n"
        "5. **Rate overall completeness** — what's the single biggest missing piece?\n\n"
        f"ORIGINAL QUESTION: {question}\n\n"
        f"AGENT OUTPUTS:\n{round1_text}\n\n"
        "Provide a structured critique with specific, actionable recommendations "
        "for improving the final answer. Be critical — this is peer review, not praise."
    )

    critique_resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": critique_prompt}],
        temperature=0.3,  # Slightly creative for finding blind spots
    )
    critique_text = critique_resp.choices[0].message.content or ""

    trace.append({
        "phase": "tool_result",
        "message": "✅ Cross-review critique complete",
        "detail": f"Critique length: {len(critique_text)} chars",
        "elapsed_ms": _elapsed(),
    })

    # ── Round 3: Deep synthesis with critique ──────────────────
    trace.append({
        "phase": "synthesis",
        "message": "✍️ Round 3/3: Deep synthesis — integrating all rounds & critique…",
        "detail": "Incorporating agent outputs + peer-review critique",
        "elapsed_ms": _elapsed(),
    })

    deep_synthesis_prompt = (
        "You are LifeScienceBench, an expert scientific research synthesizer performing "
        "a DEEP multi-round analysis. You have TWO information sources:\n\n"
        "**SOURCE A — Agent Outputs (Round 1):**\n"
        f"{round1_text}\n\n"
        "**SOURCE B — Peer-Review Critique (Round 2):**\n"
        f"{critique_text}\n\n"
        f"**ORIGINAL QUESTION:** {question}\n\n"
        "Your task: Produce the DEFINITIVE answer that:\n"
        "1. Integrates ALL agent perspectives into a unified narrative\n"
        "2. Addresses EVERY gap and conflict identified in the critique\n"
        "3. Clearly distinguishes well-established facts from emerging/uncertain findings\n"
        "4. Provides specific data, mechanisms, and citations where available\n"
        "5. Uses markdown with tables, comparison sections, and structured headings\n"
        "6. Includes a 'Limitations & Uncertainties' section at the end\n"
        "7. Is comprehensive — think of this as a mini-review article, not a quick answer\n\n"
        "Be thorough. This is the final answer after deep multi-round analysis."
    )

    deep_resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": deep_synthesis_prompt}],
        temperature=0.0,
        max_tokens=4096,
    )
    answer = deep_resp.choices[0].message.content or "No synthesis generated."

    # ── Deep analysis footer ───────────────────────────────────
    footer = "\n\n---\n## 🔬 Deep Analysis Trace (3-Round)\n"
    footer += f"**{len(agent_results)} agents** × **3 rounds** = deep synthesis in {_elapsed()/1000:.1f}s\n\n"
    footer += "### Round 1 — Agent Outputs\n"
    for agent_name, result in agent_results.items():
        if isinstance(result, dict):
            data = result.get("data", {}) if isinstance(result.get("data"), dict) else {}
            conf = data.get("confidence", "") or result.get("confidence", "N/A")
            refs = data.get("references", "") or ""
            footer += f"- **{agent_name}**: confidence={conf}"
            if refs:
                footer += f", refs={refs[:200]}"
            footer += "\n"
    footer += f"\n### Round 2 — Peer-Review Critique\n{critique_text[:1500]}{'…' if len(critique_text) > 1500 else ''}\n"
    footer += "\n### Round 3 — Deep Synthesis\n✅ Final answer above incorporates all rounds.\n"

    answer += footer

    trace.append({
        "phase": "done",
        "message": "✅ Deep 3-round analysis complete",
        "detail": f"Total time: {_elapsed()}ms, {len(agent_results)} agents × 3 rounds",
        "elapsed_ms": _elapsed(),
    })

    return answer, trace, agent_messages


# ── Streaming helper ────────────────────────────────────────────

async def _stream_sse(events):
    """Yield Server-Sent Events from a list of event dicts."""
    for event in events:
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0.01)  # Small yield to push chunks

async def _stream_json_response(markdown: str, trace: list[dict], agent_messages: list[str] | None = None):
    """Stream a JSON response as SSE chunks for real-time rendering in the pipe."""
    import time
    # First, emit all trace events
    for step in trace:
        yield f"data: {json.dumps({'type': 'status', 'message': step.get('message', ''), 'done': step.get('phase') == 'done'})}\n\n"
        await asyncio.sleep(0.005)

    # Emit agent messages
    if agent_messages:
        for am in agent_messages:
            yield f"data: {json.dumps({'type': 'status', 'message': am, 'done': False})}\n\n"
            await asyncio.sleep(0.005)

    # Stream markdown content in chunks
    if markdown:
        chunk_size = 16
        for i in range(0, len(markdown), chunk_size):
            chunk = markdown[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
            await asyncio.sleep(0.003)

    # Done signal
    yield f"data: {json.dumps({'type': 'done', 'message': '✅ Complete'})}\n\n"


# ── OpenAI-compatible /v1/chat/completions (SSE streaming) ───────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "lifesciencebench-research"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with streaming."""
    # Extract the user question from messages
    question = ""
    for msg in req.messages:
        if msg.role == "user":
            question += msg.content + "\n"

    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    if req.stream:
        from fastapi.responses import StreamingResponse

        async def generate():
            import time
            import asyncio
            t0 = time.time()
            chunk_id = f"chatcmpl-{uuid4().hex[:12]}"

            # Planning phase
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(t0), 'model': 'lifesciencebench-research', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"

            # Run the actual research
            try:
                markdown, trace = _run_fast(question)
            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'lifesciencebench-research', 'choices': [{'index': 0, 'delta': {'content': f'Error: {str(e)}'}, 'finish_reason': 'error'}]})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Stream the markdown in chunks
            chunk_size = 12
            for i in range(0, len(markdown), chunk_size):
                chunk = markdown[i:i + chunk_size]
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'lifesciencebench-research', 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01)

            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'lifesciencebench-research', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        try:
            markdown, trace = _run_fast(question)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "id": f"chatcmpl-{uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": 1720000000,
            "model": "lifesciencebench-research",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": markdown},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


# ── OpenAI-compatible /v1/models endpoint ────────────────────────

@app.get("/v1/models")
def list_models():
    """OpenAI-compatible models list so Open WebUI can discover the bridge."""
    return {
        "object": "list",
        "data": [
            {
                "id": "lifesciencebench-research",
                "object": "model",
                "created": 1720000000,
                "owned_by": "langgraph-bridge",
            }
        ],
    }


# ── Overview endpoint ───────────────────────────────────────────

@app.get("/overview")
def overview():
    """Return LangGraph agent architecture overview."""
    agents = []
    for name in list_agent_names():
        meta = get_agent(name)
        agents.append({
            "name": name,
            "domain": meta.domain if meta else "unknown",
            "description": meta.description if meta else "",
            "tools": meta.tool_names if meta else [],
        })

    tools = []
    for tname in list_tool_names():
        tmeta = get_tool(tname)
        tools.append({
            "name": tname,
            "domain": tmeta.domain if tmeta else "",
            "description": tmeta.description if tmeta else "",
            "requires_input": tmeta.requires_input if tmeta else [],
        })

    return {
        "architecture": "LangGraph Supervisor → Domain Agents → Tools",
        "supervisor_llm": _get_model_name(),
        "provider": LLM_PROVIDER,
        "agents": agents,
        "tools": tools,
        "agent_tool_map": AGENT_TOOL_MAP,
        "capabilities": [
            "multi_agent_dispatch",
            "llm_function_calling",
            "sse_streaming",
            "hybrid_retrieval",
            "structured_ingest",
            "provenance_tracking",
        ],
    }


# ── LangGraph Visualization Endpoint ──────────────────────────────

@app.get("/graph")
def langgraph_graph():
    """Return an interactive HTML page showing the actual LangGraph execution flows."""
    from fastapi.responses import HTMLResponse

    # Dynamically build agent rows
    agents_html = ""
    for name in list_agent_names():
        meta = get_agent(name)
        tools_list = ", ".join(meta.tool_names) if meta and meta.tool_names else "none"
        agents_html += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{meta.domain if meta else 'unknown'}</td>
            <td>{meta.description if meta else ''}</td>
            <td><code>{tools_list}</code></td>
        </tr>"""

    tools_html = ""
    for tname in list_tool_names():
        tmeta = get_tool(tname)
        tools_html += f"""
        <tr>
            <td><code>{tname}</code></td>
            <td>{tmeta.domain if tmeta else ''}</td>
            <td>{tmeta.description if tmeta else ''}</td>
        </tr>"""

    # Dynamically build Mermaid agent nodes
    agent_icons = {
        "chemistry": "⚗️", "biology": "🧬", "medical": "🏥",
        "literature_review": "📚", "drug_discovery": "💊", "bioinformatics": "🖥️",
        "deep_research": "🔬", "scientific_writer": "✍️",
        "statistics_advisor": "📊", "research_planner": "🗺️",
    }
    agent_nodes = ""
    agent_to_disc = ""
    for name in list_agent_names():
        icon = agent_icons.get(name, "🤖")
        display = name.replace("_", " ").title()
        tool_count = len(AGENT_TOOL_MAP.get(name, []))
        agent_nodes += f'    S --> {name.upper()}["{icon} {display}<br/><small>{tool_count} tools</small>"]\n'
        agent_to_disc += f"    {name.upper()} --> DISC\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LifeScienceBench — Execution Architecture</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #00d4aa; border-bottom: 2px solid #00d4aa33; padding-bottom: 10px; }}
  h2 {{ color: #7ec8e3; margin-top: 30px; }}
  h3 {{ color: #e94560; margin-top: 25px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #16213e; border-radius: 8px; overflow: hidden; }}
  th {{ background: #0f3460; padding: 10px 12px; text-align: left; color: #00d4aa; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #0f3460; }}
  code {{ background: #0f3460; padding: 2px 6px; border-radius: 3px; color: #e94560; font-size: 12px; }}
  .mermaid {{ background: #16213e; padding: 20px; border-radius: 8px; margin: 15px 0; text-align: center; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
  .badge-green {{ background: #00d4aa33; color: #00d4aa; }}
  .badge-blue {{ background: #7ec8e333; color: #7ec8e3; }}
  .badge-red {{ background: #e9456033; color: #e94560; }}
  .badge-yellow {{ background: #f0c04033; color: #f0c040; }}
  .flow-box {{ background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 15px 20px; margin: 15px 0; }}
  .flow-box h3 {{ margin-top: 0; }}
  .endpoint {{ font-family: monospace; background: #0f3460; padding: 2px 8px; border-radius: 4px; color: #00d4aa; }}
  ol li {{ margin: 8px 0; }}
</style>
</head>
<body>
<h1>🧪 LifeScienceBench — Execution Architecture</h1>

<p>This page reflects the <strong>actual execution paths</strong> called by the Open WebUI pipe.
All agent and tool data is pulled live from the running LangGraph service.</p>

<!-- ═══════════════════════════════════════════════════════════════ -->
<h2>⚡ Execution Paths</h2>

<div class="flow-box">
  <h3>🚀 FAST Mode — <span class="endpoint">POST /runs</span></h3>
  <p>LLM function-calling. 1–3 LLM calls, 5–30s. Best for quick questions.</p>
  <ol>
    <li><strong>Planning:</strong> LLM decides which tool(s) to call based on the question</li>
    <li><strong>Tool Execution:</strong> Selected tools run (knowledge tools, data processing, or hybrid retrieval)</li>
    <li><strong>Synthesis:</strong> LLM composes final answer from tool results</li>
  </ol>
</div>

<div class="flow-box">
  <h3>🧠 DEEP Mode — <span class="endpoint">POST /runs/full</span></h3>
  <p>Three-round multi-agent deep analysis. 60–180s. Activated by <code>Deep Thinking</code> valve or <code>deep </code> prefix.</p>
  <ol>
    <li><strong>Round 1 — Agent Dispatch:</strong> All active agents answer independently in parallel</li>
    <li><strong>Round 2 — Cross-Review:</strong> LLM peer-reviews all outputs — identifies conflicts, gaps & blind spots</li>
    <li><strong>Round 3 — Deep Synthesis:</strong> Final answer integrating all agent perspectives + critique</li>
  </ol>
</div>

<!-- ═══════════════════════════════════════════════════════════════ -->
<h2>🤖 Domain Agents ({len(list_agent_names())})</h2>
<table>
  <tr><th>Agent</th><th>Domain</th><th>Description</th><th>Tools</th></tr>
  {agents_html}
</table>

<!-- ═══════════════════════════════════════════════════════════════ -->
<h2>🔧 Tools ({len(list_tool_names())})</h2>
<table>
  <tr><th>Tool</th><th>Domain</th><th>Description</th></tr>
  {tools_html}
</table>

<!-- ═══════════════════════════════════════════════════════════════ -->
<h2>📊 DEEP Mode Execution Graph (3-Round)</h2>
<div class="mermaid">
graph TD
    U["👤 User Question"] --> P["🔌 Pipe (OWUI)"]
    P -->|deep=True| FULL["/runs/full"]
    P -->|deep=False| FAST["/runs (LLM func-call)"]

    FULL --> R1["🔍 Round 1<br/>Parallel Agent Dispatch"]
{agent_nodes}
    {agent_to_disc}
    R1 --> R2["🔍 Round 2<br/>Cross-Review Critique"]
    R2 --> R3["✍️ Round 3<br/>Deep Synthesis"]
    R3 --> RESP["📤 Final Answer"]

    FAST --> LLM["🧠 LLM Function Calling"]
    LLM --> TOOLS["🔧 Tool Execution"]
    TOOLS --> SYNTH["✍️ Synthesis"]
    SYNTH --> RESP

    style U fill:#0f3460,stroke:#00d4aa,color:#fff
    style FULL fill:#e94560,stroke:#e94560,color:#fff
    style FAST fill:#f0c04033,stroke:#f0c040,color:#f0c040
    style R1 fill:#0f3460,stroke:#7ec8e3,color:#7ec8e3
    style R2 fill:#0f3460,stroke:#7ec8e3,color:#7ec8e3
    style R3 fill:#0f3460,stroke:#00d4aa,color:#00d4aa
    style RESP fill:#0f3460,stroke:#00d4aa,color:#00d4aa
    style DISC fill:#00d4aa33,stroke:#00d4aa,color:#00d4aa
    style LLM fill:#0f3460,stroke:#7ec8e3,color:#7ec8e3
    style TOOLS fill:#0f3460,stroke:#7ec8e3,color:#7ec8e3
    style SYNTH fill:#0f3460,stroke:#7ec8e3,color:#7ec8e3
</div>

<p style="text-align:center; color: #7ec8e3; margin-top: 20px;">
  <span class="badge badge-green">Provider: {LLM_PROVIDER}</span>
  <span class="badge badge-blue">Model: {_get_model_name()}</span>
  <span class="badge badge-red">{len(list_agent_names())} agents</span>
  <span class="badge badge-green">{len(list_tool_names())} tools</span>
  <span class="badge badge-yellow">Port: 8010</span>
</p>
</body>
</html>"""
    return HTMLResponse(content=html)


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    agents = list_agent_names()
    tools = list_tool_names()
    model = _get_model_name()
    return {
        "status": "ok",
        "release": "0.3.2-langgraph",
        "provider": LLM_PROVIDER,
        "model": model,
        "agents": agents,
        "tools": tools,
    }


@app.post("/runs")
async def run(request: RunRequest):
    """
    Fast research: LLM function-calling or multi-agent dispatch.
    Uses force_multi_agent flag to decide dispatch strategy.
    """
    thread_id = str(uuid4())
    try:
        print(f"\n{'='*60}")
        print(f"📥 /runs — question: {request.question[:100]}...")
        print(f"📥 active_agents: {request.active_agents}, force_multi: {request.force_multi_agent}")
        print(f"{'='*60}")

        if request.force_multi_agent and request.active_agents and len(request.active_agents) > 1:
            markdown, trace, agent_messages = _run_multi_agent(
                request.question, request.active_agents
            )
        else:
            markdown, trace = _run_fast(request.question, request.active_agents)
            agent_messages = []

        return {
            "markdown": markdown,
            "mode": "multi" if request.force_multi_agent else "fast",
            "run_id": thread_id,
            "trace": trace,
            "agent_messages": agent_messages,
        }
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"Run error: {exc}\n{tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Execution failed: {str(exc)}",
        )


@app.post("/runs/full")
async def run_full(request: RunRequest):
    """
    Full LangGraph multi-agent dispatch (deep thinking mode).
    Calls ALL active agents, then synthesizes with discussion approach.
    """
    thread_id = str(uuid4())
    try:
        print(f"\n{'='*60}")
        print(f"📥 /runs/full (DEEP) — question: {request.question[:100]}...")
        print(f"📥 active_agents: {request.active_agents}")
        print(f"{'='*60}")

        if not request.active_agents:
            request.active_agents = ["chemistry", "biology", "medical", "literature_review"]

        markdown, trace, agent_messages = _run_multi_agent(
            request.question, request.active_agents, deep=True
        )

        return {
            "markdown": markdown,
            "mode": "full",
            "run_id": thread_id,
            "trace": trace,
            "agent_messages": agent_messages,
        }
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"Run error: {exc}\n{tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Execution failed: {str(exc)}",
        )


# ── Run standalone ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("langgraph_app.service:app", host="127.0.0.1", port=8010, reload=False)
