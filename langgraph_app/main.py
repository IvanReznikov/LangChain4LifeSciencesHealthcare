#!/usr/bin/env python3
"""
LangGraph LifeScienceBench — Terminal Application
=================================================
A LangGraph-powered scientific research assistant for chemistry, biology,
healthcare, literature review, and multi-perspective discussion.

Usage:
    python -m langgraph_app.main
    python -m langgraph_app.main --mode auto
    python -m langgraph_app.main --mode interactive

Environment:
    LC4LSH_OPENAI_API_KEY   OpenAI API key (or OPENAI_API_KEY)
    LC4LSH_MODEL_ID         Model ID (default: gpt-4o-mini)
    LC4LSH_LLM_PROVIDER     openai | groq | anthropic | ollama
    LANGCHAIN_API_KEY       LangSmith tracing (optional)

The graph discovers agents & tools on startup.
You choose which agents to use for each question.
"""

import sys
import os
import argparse
import textwrap
from typing import Optional
from uuid import uuid4

# ── Ensure project root is on path ────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Trigger tool & agent registration via imports ──────────────
# The side-effect of importing these modules is that they call register_*()
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

from langgraph_app.agents.registry import list_agent_names, get_agent, get_agents_by_domain
from langgraph_app.tools.registry import list_tool_names, get_tools_by_domain
from langgraph_app.graph import get_graph
from langgraph_app.config import LLM_PROVIDER, MODEL_ID


# ── Banner ─────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║     LifeScienceBench — Supervisor-Agent Architecture        ║
║     🧠 Supervisor (DeepSeek) + ⚗ Chemistry · 🧬 Biology    ║
║     🏥 Medical · 📚 Literature · 🗣 Discussion Board       ║
║     💊 Drug Discovery · 🖥 Bioinformatics                 ║
║     🔬 Deep Research · ✍ Scientific Writer                ║
║     📊 Statistics · 🗺 Research Planner                   ║
║     Multi-Team Research with Tool-Calling Agents            ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_discovery():
    """Print discovered agents and tools."""
    agents = list_agent_names()
    tools = list_tool_names()
    agents_by_domain = get_agents_by_domain()
    tools_by_domain = get_tools_by_domain()

    print(f"\n{'='*60}")
    print("SUPERVISOR-AGENT ARCHITECTURE")
    print(f"{'='*60}")

    from langgraph_app.config import DEEPSEEK_PRO_MODEL, LLM_PROVIDER, MODEL_ID
    model_str = DEEPSEEK_PRO_MODEL if LLM_PROVIDER == "deepseek" else MODEL_ID
    print(f"\n  🧠 Supervisor: {LLM_PROVIDER}/{model_str}")
    print(f"     Plans routing → dispatches teams → reviews results")

    print(f"\n  Teams: {len(agents)}  |  Tools: {len(tools)}")

    print(f"\n  ┌─ TEAMS ───────────────────────────────────┐")
    for domain, agent_list in agents_by_domain.items():
        icon = {"chemistry": "⚗", "biology": "🧬", "medical": "🏥", "literature": "📚", "discussion": "🗣"}.get(domain, "•")
        print(f"  │ {icon} {domain.upper():<12} → {', '.join(agent_list)}")
    print(f"  └────────────────────────────────────────────┘")

    print(f"\n  ┌─ TOOLS ───────────────────────────────────┐")
    for domain, tool_list in tools_by_domain.items():
        print(f"  │ {domain.upper():<15} → {', '.join(tool_list)}")
    print(f"  └────────────────────────────────────────────┘")
    print()


def print_agent_details():
    """Print detailed descriptions for each team."""
    print(f"\n{'='*60}")
    print("TEAM DETAILS (LLM-powered with tool access)")
    print(f"{'='*60}")
    for name in list_agent_names():
        meta = get_agent(name)
        if meta:
            tools = ', '.join(meta.tool_names) if meta.tool_names else 'none'
            llm_flag = "🔮 LLM-powered" if meta.requires_llm else "⚡ Direct tools"
            safe_flag = "" if meta.is_safe_for_auto else " ⚠ Gated"
            print(f"\n  [{name}]{safe_flag} {llm_flag}")
            print(f"    {meta.description}")
            print(f"    Tools: {tools}")
    print()


def run_query(question: str, selected_agents: Optional[list[str]] = None) -> str:
    """
    Run a single query through the Supervisor-Agent LangGraph.
    Returns the final report as a string.
    """
    graph = get_graph()

    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": question,
        "selected_agents": selected_agents or [],
        "iteration_count": 0,
        "error": None,
    }

    print(f"\n{'─'*60}")
    print(f"Question: {question[:80]}{'...' if len(question) > 80 else ''}")
    if selected_agents:
        print(f"Teams: {', '.join(selected_agents)}")
    print(f"{'─'*60}\n")

    final_state = None
    seen_outputs: set[str] = set()
    team_icons = {
        "chemistry": "⚗", "biology": "🧬", "medical": "🏥",
        "literature_review": "📚", "discussion": "🗣",
    }

    try:
        for event in graph.stream(initial_state, config, stream_mode="values"):
            node_name = event.get("next_action", "")

            # Show supervisor thought
            thought = event.get("supervisor_thought")
            if thought and thought not in seen_outputs:
                seen_outputs.add(thought)
                print(f"  🧠 Supervisor: {thought[:120]}")

            # Show team results (de-duplicated)
            agents = event.get("agent_outputs", {})
            for aname, aout in agents.items():
                if aname == "__pending__":
                    continue
                output_key = f"{aname}:{aout.get('summary', '')}"
                if output_key not in seen_outputs:
                    seen_outputs.add(output_key)
                    icon = team_icons.get(aname, "•")
                    status_icon = {"success": "✅", "error": "❌", "blocked": "🚫", "abstain": "⚠️"}.get(
                        aout.get("status", ""), "•"
                    )
                    print(f"  {icon} {status_icon} {aname}: {aout.get('summary', '')[:120]}")
                    # Show LLM reasoning snippet if available
                    llm_reasoning = aout.get("llm_reasoning", "")
                    if llm_reasoning:
                        print(f"      └─ {llm_reasoning[:150]}")

            if node_name == "stop":
                if event.get("error"):
                    print(f"  🚫 {event['error']}")
            final_state = event
    except Exception as e:
        print(f"\n  ❌ Graph error: {e}")
        return f"Error: {e}"

    if final_state:
        return final_state.get("final_report", "No report generated.")
    return "No output produced."


def interactive_mode():
    """Interactive REPL: user types queries, selects agents, gets reports."""
    print(BANNER)
    print_discovery()
    print_agent_details()

    print(f"\n{'='*60}")
    print("COMMANDS")
    print(f"{'='*60}")
    print("  /agents          List available teams with details")
    print("  /tools           List available tools by domain")
    print("  /use <a1,a2,...> Select teams to use (e.g., /use chemistry,biology)")
    print("  /use none        Let supervisor auto-select teams")
    print("  /auto            Auto-detect (supervisor plans routing)")
    print("  /help            Show this help")
    print("  /quit            Exit")
    print()
    print("  Just type a research question — the supervisor will route it!")
    print("  Questions can be open-ended: the LLM-powered teams use tools as needed.")
    print(f"{'='*60}\n")

    selected_agents: list[str] = []
    use_auto = True  # Default: auto-detect

    while True:
        try:
            if use_auto:
                prompt = "🔬 [auto] > "
            elif selected_agents:
                prompt = f"🔬 [{' + '.join(selected_agents)}] > "
            else:
                prompt = "🔬 > "

            user_input = input(prompt).strip()

            if not user_input:
                continue

            # ── Commands ──
            if user_input.startswith("/"):
                cmd, *args = user_input.split(maxsplit=1)
                arg = args[0] if args else ""

                if cmd == "/quit" or cmd == "/exit":
                    print("\nGoodbye! 👋\n")
                    break

                elif cmd == "/help":
                    print("\nCOMMANDS:")
                    print("  /agents          List available teams")
                    print("  /tools           List available tools")
                    print("  /use <a1,a2,...> Select teams (e.g., /use chemistry,literature_review)")
                    print("  /use none        Let supervisor auto-plan routing")
                    print("  /auto            Auto-detect (supervisor plans routing)")
                    print("  /help            Show this help")
                    print("  /quit            Exit\n")

                elif cmd == "/agents":
                    print_agent_details()

                elif cmd == "/tools":
                    tools_by_domain = get_tools_by_domain()
                    print(f"\n{'='*60}")
                    print("AVAILABLE TOOLS")
                    print(f"{'='*60}")
                    for domain, tool_list in tools_by_domain.items():
                        print(f"\n  {domain.upper()}:")
                        for tname in tool_list:
                            from langgraph_app.tools.registry import get_tool
                            tmeta = get_tool(tname)
                            if tmeta:
                                print(f"    • {tname}: {tmeta.description[:100]}")
                    print()

                elif cmd == "/use":
                    if not arg or arg.lower() == "none":
                        selected_agents = []
                        use_auto = True
                        print("  → Supervisor will auto-plan team routing")
                    else:
                        requested = [a.strip() for a in arg.split(",")]
                        valid_agents = list_agent_names()
                        selected_agents = [a for a in requested if a in valid_agents]
                        invalid = [a for a in requested if a not in valid_agents]
                        if invalid:
                            print(f"  ⚠ Unknown: {', '.join(invalid)}")
                            print(f"  Valid: {', '.join(valid_agents)}")
                        if selected_agents:
                            use_auto = False
                            print(f"  → Using teams: {', '.join(selected_agents)}")

                elif cmd == "/auto":
                    selected_agents = []
                    use_auto = True
                    print("  → Supervisor auto-planning enabled")

                else:
                    print(f"  Unknown command: {cmd}. Type /help for available commands.")

            else:
                # ── Execute query ──
                agents_to_use = None if use_auto else (selected_agents or None)
                report = run_query(user_input, agents_to_use)

                print(f"\n{'='*60}")
                print("FINAL REPORT")
                print(f"{'='*60}\n")
                # Print with word-wrap for terminal readability
                for line in report.split('\n'):
                    print(textwrap.fill(line, width=100, subsequent_indent='  ') if len(line) > 100 else line)
                print(f"\n{'='*60}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.\n")
        except EOFError:
            print("\nGoodbye! 👋\n")
            break


def auto_mode(question: Optional[str] = None):
    """One-shot auto mode: run a single question or show a demo."""
    print(BANNER)
    print_discovery()

    if question:
        report = run_query(question)
        print(f"\n{'='*60}")
        print("FINAL REPORT")
        print(f"{'='*60}\n")
        print(report)
        print(f"\n{'='*60}\n")
    else:
        # Demo with a sample query
        demo_questions = [
            "SMILES: CCO — identify this molecule",
            "Analyze the protein sequence ACDEFGHIKLMNPQRSTVWYACDEFGHIKL",
            "Review literature on ethanol toxicity",
        ]
        print("\nDEMO MODE — Running sample queries:\n")
        for q in demo_questions:
            print(f"\n{'#'*60}")
            print(f"# {q}")
            print(f"{'#'*60}")
            report = run_query(q)
            print(report[:500] + ("..." if len(report) > 500 else ""))


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LifeScienceBench — LangGraph Scientific Research Assistant"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["interactive", "auto"],
        default="interactive",
        help="Run mode: interactive REPL or auto one-shot (default: interactive)",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Single question for auto mode",
    )
    parser.add_argument(
        "--agents", "-a",
        type=str,
        default=None,
        help="Comma-separated agent names (e.g., 'chemistry,literature_review')",
    )
    args = parser.parse_args()

    if args.mode == "auto":
        selected = None
        if args.agents:
            selected = [a.strip() for a in args.agents.split(",")]
            report = run_query(args.question or "SMILES: CCO — identify this molecule", selected)
            print(report)
        else:
            auto_mode(args.question)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
