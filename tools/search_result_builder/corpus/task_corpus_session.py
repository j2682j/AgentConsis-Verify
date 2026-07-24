from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.network_utils import normalize_text

from .jsonl_exporter import JSONLExporter
from .web_corpus_builder import CorpusRecord


class TaskCorpusSession:
    """Own one task's mutable corpus and incrementally extend its FAISS index."""

    def __init__(
        self,
        *,
        corpus_path: str | Path,
        retriever: Any,
        exporter: JSONLExporter | None = None,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.retriever = retriever
        self.exporter = exporter or JSONLExporter()
        self._known_hashes = {
            self._content_hash(str(document.get("text") or ""))
            for document in retriever.passage_map.values()
            if self._content_hash(str(document.get("text") or ""))
        }
        self._hop_index = 0

    def add_records(self, records: Iterable[CorpusRecord]) -> list[CorpusRecord]:
        """Append unseen passages and add only their new vectors to the live index."""
        self._hop_index += 1
        accepted: list[CorpusRecord] = []
        for record in records:
            content_hash = self._content_hash(record.text)
            if not content_hash or content_hash in self._known_hashes:
                continue
            accepted.append(
                replace(
                    record,
                    id=f"hop-{self._hop_index:02d}-{len(accepted) + 1:04d}",
                )
            )
            self._known_hashes.add(content_hash)
        if not accepted:
            return []

        self.exporter.export(accepted, self.corpus_path, append=True)
        payloads = [record.to_dict() for record in accepted]
        passage_texts = [
            self.retriever.embedder.prepare_passage_text(payload)
            for payload in payloads
        ]
        embeddings = self.retriever.embedder.embed(passage_texts).astype("float32")
        index = self.retriever.index.index
        if not index.is_trained:
            index.train(embeddings)
        index.add(np.asarray(embeddings, dtype="float32"))
        ids = [record.id for record in accepted]
        self.retriever.index.idx2db.extend(ids)
        self.retriever.passage_map.update(
            {record.id: record.to_dict() for record in accepted}
        )
        rebuild_page_index = getattr(self.retriever, "rebuild_page_index", None)
        if callable(rebuild_page_index):
            rebuild_page_index()
        return accepted

    def _content_hash(self, text: str) -> str:
        cleaned = normalize_text(text).casefold()
        if not cleaned:
            return ""
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


__all__ = ["TaskCorpusSession"]
