"""TF-IDF + cosine retrieval over runbook sections (numpy only)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_STOP = {"a", "an", "the", "and", "or", "of", "to", "on", "in", "is", "as", "at", "for", "be", "by"}


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens without stop words."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    """One runbook section."""

    runbook_id: str
    title: str
    section: str
    text: str
    meta: dict[str, str]


def load_runbooks(directory: Path) -> list[Chunk]:
    """Parse `*.md` runbooks into per-section chunks (`## ` headings)."""
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        m = re.match(r"#\s+(\S+?):\s*(.+)", lines[0]) if lines else None
        if not m:
            continue
        rid, title = m.group(1), m.group(2).strip()
        meta: dict[str, str] = {}
        section, buf = "Overview", []
        parts: list[tuple[str, str]] = []
        for ln in lines[1:]:
            mm = re.match(r"-\s*(\w+):\s*(.+)", ln)
            if mm and section == "Overview":
                meta[mm.group(1)] = mm.group(2).strip()
            elif ln.startswith("## "):
                parts.append((section, " ".join(buf).strip()))
                section, buf = ln[3:].strip(), []
            else:
                buf.append(ln.strip())
        parts.append((section, " ".join(buf).strip()))
        for sec, body in parts:
            if body:
                chunks.append(Chunk(rid, title, sec, body, meta))
    return chunks


class RunbookIndex:
    """TF-IDF index (sublinear tf, smoothed idf, l2 norm) with cosine search."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        docs = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        vocab = sorted({t for d in docs for t in d})
        self.vocab = {t: i for i, t in enumerate(vocab)}
        df = np.zeros(len(vocab))
        for d in docs:
            for t in set(d):
                df[self.vocab[t]] += 1
        n = max(len(docs), 1)
        self.idf = np.log((1 + n) / (1 + df)) + 1.0
        self.matrix = np.vstack([self._vec(d) for d in docs]) if docs else np.zeros((0, 0))

    def _vec(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in tokens:
            if t in self.vocab:
                v[self.vocab[t]] += 1
        v = np.where(v > 0, 1 + np.log(np.maximum(v, 1)), 0.0) * self.idf
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Top-k sections by cosine similarity: runbook_id, title, section, excerpt, score."""
        if not self.chunks:
            return []
        sims = self.matrix @ self._vec(tokenize(query))
        order = np.argsort(-sims, kind="stable")[:k]
        return [
            {
                "runbook_id": self.chunks[i].runbook_id,
                "title": self.chunks[i].title,
                "section": self.chunks[i].section,
                "excerpt": self.chunks[i].text,
                "score": round(float(sims[i]), 4),
                "action": self.chunks[i].meta.get("action"),
                "risk": self.chunks[i].meta.get("risk"),
            }
            for i in order
            if math.isfinite(sims[i])
        ]
