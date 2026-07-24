"""
LangGraph LifeScienceBench — Agent State schema.

CRITICAL LangGraph lessons (from repo memory):
1. ALL keys used across nodes must be declared in the TypedDict.
2. Conditional edge functions are READ-ONLY — mutations are NOT persisted.
3. Use `lambda old, new: new` reducer for lists when nodes do in-place accumulation.
"""

from typing import TypedDict, Annotated, Any, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── User input ────────────────────────────────────────────
    question: str
    messages: Annotated[list, add_messages]  # Chat history

    # ── Discovery & planning ──────────────────────────────────
    available_agents: list[str]        # Agent names discovered on startup
    available_tools: list[str]         # Tool names discovered on startup
    selected_agents: list[str]         # Supervisor-chosen agents for this run
    selected_tools: list[str]          # User-chosen tools for this run
    mode: str                          # auto | manual | board | review

    # ── Supervisor ────────────────────────────────────────────
    supervisor_plan: dict[str, Any]    # Supervisor's routing plan: {agent: sub_question, ...}
    supervisor_thought: str            # Supervisor's reasoning trace
    supervisor_iteration: int          # How many times supervisor has reviewed

    # ── Execution ─────────────────────────────────────────────
    agent_outputs: Annotated[dict[str, Any], lambda old, new: new]  # agent_name → output
    tool_outputs: Annotated[dict[str, Any], lambda old, new: new]   # tool_name → output

    # ── RAG evidence ──────────────────────────────────────────
    evidence: list[dict[str, Any]]     # Retrieved evidence chunks
    retrieval_status: str              # success | partial | none

    # ── Synthesis & review ────────────────────────────────────
    synthesis: str                     # Supervisor's combined synthesis
    board_review: str                  # Discussion panel output
    final_report: str                  # Final markdown report

    # ── Control flow ──────────────────────────────────────────
    next_action: str                   # continue | stop | error | loop
    error: Optional[str]               # Error message if any
    iteration_count: int               # Safety counter
