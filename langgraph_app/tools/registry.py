"""
LangGraph LifeScienceBench — Tool Registry.
Dynamic tool discovery: tools register themselves with metadata.
The graph learns what tools are available at startup.
"""

from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class ToolMeta:
    """Metadata for a discoverable tool."""
    name: str
    description: str
    domain: str                     # chemistry | biology | medical | rag | visualization | curation
    requires_input: list[str] = field(default_factory=list)  # e.g. ["smiles", "csv_text"]
    produces: str = "artifact"      # artifact | figure | evidence | data
    requires_llm: bool = False      # Does this tool need an LLM call?
    is_dangerous: bool = False      # Should be gated (e.g., medical diagnosis)
    func: Optional[Callable] = None # The actual callable


# ── Global registry ───────────────────────────────────────────
_tool_registry: dict[str, ToolMeta] = {}


def register_tool(meta: ToolMeta) -> ToolMeta:
    """Register a tool for discovery."""
    _tool_registry[meta.name] = meta
    return meta


def discover_tools(domain: Optional[str] = None) -> list[ToolMeta]:
    """Return all registered tools, optionally filtered by domain."""
    tools = list(_tool_registry.values())
    if domain:
        tools = [t for t in tools if t.domain == domain]
    return tools


def get_tool(name: str) -> Optional[ToolMeta]:
    """Get a specific tool by name."""
    return _tool_registry.get(name)


def list_tool_names() -> list[str]:
    """Return all registered tool names."""
    return list(_tool_registry.keys())


def get_tools_by_domain() -> dict[str, list[str]]:
    """Return tools grouped by domain."""
    grouped: dict[str, list[str]] = {}
    for meta in _tool_registry.values():
        grouped.setdefault(meta.domain, []).append(meta.name)
    return grouped
