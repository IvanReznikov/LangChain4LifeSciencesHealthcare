"""
LangGraph LifeScienceBench — Agent Registry.
Agents register themselves with metadata. The graph discovers them at startup.
Each agent declares which tools it can use and its domain expertise.
"""

from typing import Callable, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentMeta:
    """Metadata for a discoverable agent."""
    name: str
    description: str
    domain: str                          # chemistry | biology | medical | literature | discussion
    tool_names: list[str] = field(default_factory=list)  # Tools this agent can use
    requires_llm: bool = True            # Does this agent need an LLM?
    is_safe_for_auto: bool = True        # Can run without user explicit approval?
    run_func: Optional[Callable] = None  # The actual agent execution function


# ── Global registry ───────────────────────────────────────────
_agent_registry: dict[str, AgentMeta] = {}


def register_agent(meta: AgentMeta) -> AgentMeta:
    """Register an agent for discovery."""
    _agent_registry[meta.name] = meta
    return meta


def discover_agents(domain: Optional[str] = None) -> list[AgentMeta]:
    """Return all registered agents, optionally filtered by domain."""
    agents = list(_agent_registry.values())
    if domain:
        agents = [a for a in agents if a.domain == domain]
    return agents


def get_agent(name: str) -> Optional[AgentMeta]:
    """Get a specific agent by name."""
    return _agent_registry.get(name)


def list_agent_names() -> list[str]:
    """Return all registered agent names."""
    return list(_agent_registry.keys())


def get_agents_by_domain() -> dict[str, list[str]]:
    """Return agents grouped by domain."""
    grouped: dict[str, list[str]] = {}
    for meta in _agent_registry.values():
        grouped.setdefault(meta.domain, []).append(meta.name)
    return grouped
