"""
LangGraph LifeScienceBench — Experiment & Protocol Team (4 agents).

💡 hypothesis_generator  — Generate novel, testable hypotheses
🧪 experiment_designer   — Design experiments to validate findings
🔧 protocol_optimizer    — Improve experimental protocols
🛠️ troubleshooting       — Diagnose failed experiments

All agents use LLM-powered knowledge tools.
"""

from typing import Any, Callable
from .registry import AgentMeta, register_agent


# ═══════════════════════════════════════════════════════════════
# 1. Hypothesis Generator
# ═══════════════════════════════════════════════════════════════

def _run_hypothesis_generator(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "hypothesis_generator", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "hypothesis_generator_knowledge" in tools:
        try:
            result = tools["hypothesis_generator_knowledge"](question=question)
            output["tool_results"]["hypothesis_generator_knowledge"] = result
            output["summary"] = "Hypothesis generator: hypotheses proposed"
        except Exception as e:
            output["tool_results"]["hypothesis_generator_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Hypothesis generator failed: {e}"
    output["warnings"].append("AI-generated hypotheses require experimental validation — they are starting points, not conclusions.")
    return output

register_agent(AgentMeta(
    name="hypothesis_generator",
    description="Generate novel, testable scientific hypotheses. Suggest mechanisms that could explain observations, propose alternative models, and identify which hypothesis is most parsimonious.",
    domain="experiment",
    tool_names=["hypothesis_generator_knowledge"],
    requires_llm=True,
    run_func=_run_hypothesis_generator,
))


# ═══════════════════════════════════════════════════════════════
# 2. Experiment Designer
# ═══════════════════════════════════════════════════════════════

def _run_experiment_designer(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "experiment_designer", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "experiment_designer_knowledge" in tools:
        try:
            result = tools["experiment_designer_knowledge"](question=question)
            output["tool_results"]["experiment_designer_knowledge"] = result
            output["summary"] = "Experiment designer: experimental plan created"
        except Exception as e:
            output["tool_results"]["experiment_designer_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Experiment designer failed: {e}"
    output["warnings"].append("AI-designed experiments need expert review — consider controls, sample sizes, and feasibility.")
    return output

register_agent(AgentMeta(
    name="experiment_designer",
    description="Design experiments to validate a hypothesis or biomarker. Specify controls, sample sizes, methods, expected outcomes, and potential pitfalls. Covers in vitro, in vivo, and clinical studies.",
    domain="experiment",
    tool_names=["experiment_designer_knowledge"],
    requires_llm=True,
    run_func=_run_experiment_designer,
))


# ═══════════════════════════════════════════════════════════════
# 3. Protocol Optimizer
# ═══════════════════════════════════════════════════════════════

def _run_protocol_optimizer(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "protocol_optimizer", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "protocol_optimizer_knowledge" in tools:
        try:
            result = tools["protocol_optimizer_knowledge"](question=question)
            output["tool_results"]["protocol_optimizer_knowledge"] = result
            output["summary"] = "Protocol optimizer: optimization suggestions provided"
        except Exception as e:
            output["tool_results"]["protocol_optimizer_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Protocol optimizer failed: {e}"
    output["warnings"].append("Protocol suggestions should be tested at small scale before full implementation.")
    return output

register_agent(AgentMeta(
    name="protocol_optimizer",
    description="Improve experimental protocols — Western blot, PCR, ELISA, cell culture, chromatography, etc. Suggest reagent concentrations, incubation times, temperature optimizations, and troubleshooting steps.",
    domain="experiment",
    tool_names=["protocol_optimizer_knowledge"],
    requires_llm=True,
    run_func=_run_protocol_optimizer,
))


# ═══════════════════════════════════════════════════════════════
# 4. Troubleshooting Assistant
# ═══════════════════════════════════════════════════════════════

def _run_troubleshooting(
    question: str, tools: dict[str, Callable], state: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "agent": "troubleshooting", "status": "success",
        "summary": "", "tool_results": {}, "warnings": [],
    }
    if "troubleshooting_knowledge" in tools:
        try:
            result = tools["troubleshooting_knowledge"](question=question)
            output["tool_results"]["troubleshooting_knowledge"] = result
            output["summary"] = "Troubleshooting: diagnosis complete"
        except Exception as e:
            output["tool_results"]["troubleshooting_knowledge"] = {"status": "error", "summary": str(e)}
            output["status"] = "error"
            output["summary"] = f"Troubleshooting failed: {e}"
    output["warnings"].append("Troubleshooting advice is general — specific reagents/instruments may have unique failure modes.")
    return output

register_agent(AgentMeta(
    name="troubleshooting",
    description="Diagnose failed experiments — PCR, cloning, Western blot, cell culture, crystallization, synthesis. Identify likely causes (reagent degradation, contamination, temperature, pH) and suggest fixes.",
    domain="experiment",
    tool_names=["troubleshooting_knowledge"],
    requires_llm=True,
    run_func=_run_troubleshooting,
))
