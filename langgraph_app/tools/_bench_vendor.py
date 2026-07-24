"""
Vendored subset of lifesciencebench-v0.3.2 bench modules.
Only the functions actually used by rag_tools.py — no Pydantic, no extra deps.
"""
import math
import re
import json
import hashlib
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ── Evidence dataclass (replaces bench.core.models.Evidence) ──
@dataclass
class Evidence:
    citation: str
    document: str
    chunk: int
    excerpt: str
    score: float

# ── CORPUS path (replaces bench.core.storage) ─────────────────
CORPUS = Path('data') / 'corpus'
CORPUS.mkdir(parents=True, exist_ok=True)


# ── bench/rag/hybrid.py ───────────────────────────────────────
def _toks(x: str) -> list[str]:
    return re.findall(r'[A-Za-z0-9_-]{2,}', x.lower())


def _chunks() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for p in CORPUS.glob('*'):
        if p.is_file() and p.suffix.lower() in {'.txt', '.md', '.csv', '.json', '.fasta', '.fa', '.pdb'}:
            text = p.read_text(errors='replace')
            for i in range(0, len(text), 900):
                out.append((p.name, i // 900, text[i:i + 900]))
    return out


def retrieve_hybrid(query: str, k: int = 8, diversity: int = 2) -> list[Evidence]:
    cs = _chunks()
    q = _toks(query)
    N = max(1, len(cs))
    df = Counter(t for _, _, x in cs for t in set(_toks(x)))
    scores: list[tuple[float, str, int, str]] = []
    for name, idx, x in cs:
        tf = Counter(_toks(x))
        lex = sum((1 + math.log(tf[t])) * math.log((N + 1) / (df[t] + 1)) for t in q if tf[t])
        qa = {''.join(q)[i:i + 3] for i in range(max(0, len(''.join(q)) - 2))}
        xa = {x.lower()[i:i + 3] for i in range(max(0, len(x) - 2))}
        char = len(qa & xa) / max(1, len(qa | xa))
        if lex or char > .01:
            scores.append((lex + 2 * char, name, idx, x))
    selected: list[Evidence] = []
    counts: Counter[str] = Counter()
    for s, n, i, x in sorted(scores, reverse=True):
        if counts[n] >= diversity:
            continue
        selected.append(Evidence(citation=f'{n}:{i}', document=n, chunk=i, excerpt=x, score=round(s, 4)))
        counts[n] += 1
        if len(selected) == k:
            break
    return selected


# ── bench/rag/local.py ────────────────────────────────────────
def ingest(name: str, text: str) -> str:
    p = CORPUS / name.split('/')[-1]
    p.write_text(text)
    return p.name


# ── bench/ingest/parsers.py ───────────────────────────────────
def _digest(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()


def _text_document(name: str, text: str) -> dict[str, Any]:
    sections: list[dict[str, str]] = []
    current = 'Preamble'
    buf: list[str] = []
    for line in text.splitlines():
        if re.match(r'^#{1,6}\s+', line):
            if buf:
                sections.append({'heading': current, 'text': '\n'.join(buf).strip()})
            current = re.sub(r'^#+\s*', '', line).strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append({'heading': current, 'text': '\n'.join(buf).strip()})
    doi_match = next(iter(re.findall(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', text, re.I)), None)
    return {
        'kind': 'document', 'name': Path(name).name, 'sha256': _digest(text),
        'created_at': datetime.now(timezone.utc).isoformat(), 'sections': sections,
        'metadata': {'doi': doi_match},
    }


def _fasta(name: str, text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    h: str | None = None
    seq: list[str] = []
    for line in text.splitlines() + ['>END']:
        if line.startswith('>'):
            if h:
                records.append({'id': h.split()[0], 'description': h, 'sequence': ''.join(seq), 'length': len(''.join(seq))})
            h = line[1:].strip()
            seq = []
        else:
            seq.append(re.sub(r'\s+', '', line).upper())
    return {'kind': 'fasta', 'name': Path(name).name, 'sha256': _digest(text), 'records': records, 'metadata': {}}


def _fhir(name: str, text: str) -> dict[str, Any]:
    x = json.loads(text)
    entries = x.get('entry', []) if x.get('resourceType') == 'Bundle' else [{'resource': x}]
    resources = [e.get('resource', {}) for e in entries]
    return {
        'kind': 'fhir', 'name': Path(name).name, 'sha256': _digest(text),
        'resources': [{'resourceType': r.get('resourceType'), 'id': r.get('id')} for r in resources],
        'metadata': {'resource_counts': {k: sum(r.get('resourceType') == k for r in resources)
                                          for k in sorted({r.get('resourceType') for r in resources})}},
    }


def _pdb(name: str, text: str) -> dict[str, Any]:
    chains = sorted({line[21].strip() or '_' for line in text.splitlines() if line.startswith(('ATOM  ', 'HETATM'))})
    ligands = sorted({line[17:20].strip() for line in text.splitlines() if line.startswith('HETATM')})
    return {
        'kind': 'structure', 'name': Path(name).name, 'sha256': _digest(text),
        'metadata': {'chains': chains, 'ligands': ligands,
                     'atoms': sum(line.startswith(('ATOM  ', 'HETATM')) for line in text.splitlines())},
    }


def parse(name: str, text: str) -> dict[str, Any]:
    ext = Path(name).suffix.lower()
    if ext in {'.fa', '.fasta', '.faa'}:
        return _fasta(name, text)
    if ext in {'.json', '.fhir'}:
        return _fhir(name, text)
    if ext in {'.pdb', '.ent'}:
        return _pdb(name, text)
    return _text_document(name, text)
