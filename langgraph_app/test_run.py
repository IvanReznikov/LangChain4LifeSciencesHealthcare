"""Quick end-to-end test of the LangGraph application."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trigger registrations
import langgraph_app.tools.chemistry_tools  # noqa
import langgraph_app.tools.biology_tools    # noqa
import langgraph_app.tools.medical_tools    # noqa
import langgraph_app.tools.rag_tools        # noqa
import langgraph_app.agents.chemistry       # noqa
import langgraph_app.agents.biology         # noqa
import langgraph_app.agents.medical         # noqa
import langgraph_app.agents.literature_review  # noqa
import langgraph_app.agents.discussion      # noqa

from langgraph_app.graph import get_graph
from uuid import uuid4

graph = get_graph()

# Test 1: Chemistry agent (no LLM needed)
print("=" * 60)
print("TEST 1: Chemistry — SMILES: CCO")
print("=" * 60)

config1 = {"configurable": {"thread_id": str(uuid4())}}
state = {
    "question": "SMILES: CCO",
    "selected_agents": ["chemistry", "literature_review"],
    "iteration_count": 0,
    "error": None,
}

final = None
for event in graph.stream(state, config1, stream_mode="values"):
    final = event
    agents = event.get("agent_outputs", {})
    for aname, aout in agents.items():
        status = aout.get("status", "?")
        summary = aout.get("summary", "")
        print(f"  [{status}] {aname}: {summary[:120]}")

print("\n--- FINAL REPORT (first 600 chars) ---")
if final:
    print(final.get("final_report", "No report")[:600])

# Test 2: Biology agent
print("\n" + "=" * 60)
print("TEST 2: Biology — Protein sequence QC")
print("=" * 60)

config2 = {"configurable": {"thread_id": str(uuid4())}}
state2 = {
    "question": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
    "selected_agents": ["biology"],
    "iteration_count": 0,
    "error": None,
}

final2 = None
for event in graph.stream(state2, config2, stream_mode="values"):
    final2 = event
    agents = event.get("agent_outputs", {})
    for aname, aout in agents.items():
        status = aout.get("status", "?")
        summary = aout.get("summary", "")
        print(f"  [{status}] {aname}: {summary[:120]}")

print("\n--- FINAL REPORT (first 600 chars) ---")
if final2:
    print(final2.get("final_report", "No report")[:600])

# Test 3: Medical safety gate
print("\n" + "=" * 60)
print("TEST 3: Medical — diagnose this patient")
print("=" * 60)

config3 = {"configurable": {"thread_id": str(uuid4())}}
state3 = {
    "question": "diagnose this patient with headache",
    "selected_agents": ["medical"],
    "iteration_count": 0,
    "error": None,
}

final3 = None
for event in graph.stream(state3, config3, stream_mode="values"):
    final3 = event
    agents = event.get("agent_outputs", {})
    for aname, aout in agents.items():
        status = aout.get("status", "?")
        summary = aout.get("summary", "")
        print(f"  [{status}] {aname}: {summary[:120]}")

print("\n--- FINAL REPORT (first 400 chars) ---")
if final3:
    print(final3.get("final_report", "No report")[:400])

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
