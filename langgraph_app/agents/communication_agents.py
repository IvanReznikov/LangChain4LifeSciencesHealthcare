"""
LangGraph LifeScienceBench — Communication & Education Team (4 agents).

🎨 figure_generator     — Design publication figures and graphical abstracts
🎤 presentation_coach   — Prepare conference talks and presentations
📖 teaching_assistant   — Explain scientific concepts at any level
📓 lab_notebook         — Organize and summarize experiments

All agents use LLM-powered knowledge tools.
"""

from typing import Any, Callable
from .registry import AgentMeta, register_agent


# ═══════════════════════════════════════════════════════════════
# 1. Figure Generator
# ═══════════════════════════════════════════════════════════════

def _run_figure_generator(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "figure_generator", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "figure_generator_knowledge" in tools:
        try:
            result = tools["figure_generator_knowledge"](question=question)
            output["tool_results"]["figure_generator_knowledge"] = result
            output["summary"] = "Figure generator: design guidance provided"
        except Exception as e:
            output["tool_results"]["figure_generator_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Figure generator failed: {e}"
    output["warnings"].append(
        "AI provides figure concepts and descriptions — actual rendering requires "
        "graphics tools (BioRender, GraphPad, matplotlib, Illustrator)."
    )
    return output

register_agent(AgentMeta(
    name="figure_generator",
    description="Design publication-quality figures — graphical abstracts, mechanism diagrams, pathway schematics, data visualization layouts. Provides detailed figure descriptions and layout recommendations.",
    domain="communication",
    tool_names=["figure_generator_knowledge"],
    requires_llm=True,
    run_func=_run_figure_generator,
))


# ═══════════════════════════════════════════════════════════════
# 2. Presentation Coach
# ═══════════════════════════════════════════════════════════════

def _run_presentation_coach(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "presentation_coach", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "presentation_coach_knowledge" in tools:
        try:
            result = tools["presentation_coach_knowledge"](question=question)
            output["tool_results"]["presentation_coach_knowledge"] = result
            output["summary"] = "Presentation coach: talk outline prepared"
        except Exception as e:
            output["tool_results"]["presentation_coach_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Presentation coach failed: {e}"
    output["warnings"].append("AI-generated presentation structure — practice delivery and tailor to your audience.")
    return output

register_agent(AgentMeta(
    name="presentation_coach",
    description="Convert a paper or research project into a conference presentation. Structure a 15-minute talk, design slide flow, anticipate audience questions, and craft a compelling narrative arc.",
    domain="communication",
    tool_names=["presentation_coach_knowledge"],
    requires_llm=True,
    run_func=_run_presentation_coach,
))


# ═══════════════════════════════════════════════════════════════
# 3. Teaching Assistant
# ═══════════════════════════════════════════════════════════════

def _run_teaching_assistant(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "teaching_assistant", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "teaching_assistant_knowledge" in tools:
        try:
            result = tools["teaching_assistant_knowledge"](question=question)
            output["tool_results"]["teaching_assistant_knowledge"] = result
            output["summary"] = "Teaching assistant: explanation provided"
        except Exception as e:
            output["tool_results"]["teaching_assistant_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Teaching assistant failed: {e}"
    output["warnings"].append("Educational content — verify with textbooks and primary literature for accuracy.")
    return output

register_agent(AgentMeta(
    name="teaching_assistant",
    description="Explain scientific concepts at any level — undergraduate, graduate, or public outreach. Break down complex topics (CRISPR, CAR-T, DFT, fMRI) with analogies, diagrams, and progressive complexity.",
    domain="communication",
    tool_names=["teaching_assistant_knowledge"],
    requires_llm=True,
    run_func=_run_teaching_assistant,
))


# ═══════════════════════════════════════════════════════════════
# 4. Lab Notebook
# ═══════════════════════════════════════════════════════════════

def _run_lab_notebook(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "lab_notebook", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "lab_notebook_knowledge" in tools:
        try:
            result = tools["lab_notebook_knowledge"](question=question)
            output["tool_results"]["lab_notebook_knowledge"] = result
            output["summary"] = "Lab notebook: experiments summarized"
        except Exception as e:
            output["tool_results"]["lab_notebook_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Lab notebook failed: {e}"
    output["warnings"].append(
        "AI-generated lab notebook entries should be reviewed and supplemented "
        "with raw data, instrument files, and experimental metadata."
    )
    return output

register_agent(AgentMeta(
    name="lab_notebook",
    description="Organize and summarize experiments — structure daily notes, track reagent lots, record observations, suggest next steps, and format entries for ELN (electronic lab notebook) compliance.",
    domain="communication",
    tool_names=["lab_notebook_knowledge"],
    requires_llm=True,
    run_func=_run_lab_notebook,
))
