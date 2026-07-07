import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import TqdmExperimentalWarning, tqdm
from tqdm.rich import tqdm_rich

from .contriever import Contriever
from .e5 import (
    E5BaseV2Embedding,
    E5LargeV2Embedding,
    MultilingualE5BaseEmbedding,
)
from .utils.normalize_text import normalize

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

try:
    from .ada_embedding import AdaEmbedding
except ModuleNotFoundError:
    AdaEmbedding = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MULTILINGUAL_E5_BASE_PATH = (
    PROJECT_ROOT
    / "models"
    / "retriever"
    / "model_cache"
    / "multilingual-e5-base"
)

EmbeddingModelTypes = Literal[
    "contriever",
    "e5-base-v2",
    "e5-large-v2",
    "multilingual-e5-base",
    "e5-mistral-instruct",
    "ada-002",
]

ModelTypes = {
    "contriever": Contriever,
    "e5-base-v2": E5BaseV2Embedding,
    "e5-large-v2": E5LargeV2Embedding,
    "multilingual-e5-base": MultilingualE5BaseEmbedding,
}
if AdaEmbedding is not None:
    ModelTypes["ada-002"] = AdaEmbedding

ModelCheckpointMapping = {
    "contriever": "model_cache/contriever-msmarco",
    "e5-base-v2": "model_cache/e5-base-v2",
    "e5-large-v2": "model_cache/e5-large-v2",
    "multilingual-e5-base": str(MULTILINGUAL_E5_BASE_PATH),
    "ada-002": "text-embedding-ada-002",
}


class Embedder(object):
    def __init__(
        self,
        model_type: EmbeddingModelTypes,
        model_name_or_path: str = None,
        batch_size: int = 128,
        chunk_size: int = int(2e6),
        text_lower_case: bool = False,
        text_normalize: bool = False,
        no_title: bool = False,
    ):
        if model_name_or_path is None:
            model_name_or_path = ModelCheckpointMapping[model_type]
        self.model_type = model_type
        self.embedder = ModelTypes[model_type](model_name_or_path)
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        self.text_lower_case = text_lower_case
        self.text_normalize = text_normalize
        self.no_title = no_title

    def process_text(self, line):
        if isinstance(line, dict):
            if self.no_title or "title" not in line:
                text = line["text"]
            else:
                text = f"{line['title']}: {line['text']}"
        else:
            text = line

        if self.text_lower_case:
            text = text.lower()
        if self.text_normalize:
            text = normalize(text)
        return text

    def get_ids(self, data):
        return [line["id"] for line in data]

    def embed_passages(self, data):
        ids = self.get_ids(data)
        texts = [self.prepare_passage_text(line) for line in data]

        chunkBatch = (len(texts) - 1) // self.chunk_size + 1
        with torch.no_grad():
            for idx in range(chunkBatch):
                print(f"Processing chunk {idx + 1}/{chunkBatch}")
                chunkStartIdx = idx * self.chunk_size
                chunkEndIdx = min((idx + 1) * self.chunk_size, len(texts))
                chunk = texts[chunkStartIdx:chunkEndIdx]
                chunk_ids = ids[chunkStartIdx:chunkEndIdx]
                chunk_embeddings = self.embed(chunk, verbose=True)
                yield idx, (chunk_ids, chunk_embeddings)

    def prepare_passage_text(self, line):
        text = self.process_text(line)
        if self.model_type == "multilingual-e5-base":
            return f"passage: {text}"
        return text

    def prepare_query_text(self, text):
        text = self.process_text(text)
        if self.model_type == "multilingual-e5-base":
            return f"query: {text}"
        return text

    def embed(self, textBatch, verbose=False):
        embeddings = np.array([])
        textBatch = [self.process_text(text) for text in textBatch]
        batches = (len(textBatch) - 1) // self.batch_size + 1
        with torch.no_grad():
            if verbose:
                iter_range = tqdm_rich(range(batches), desc="Embedding")
            else:
                iter_range = range(batches)
            # for idx in tqdm_rich(range(batches), desc="Embedding"):
            for idx in iter_range:
                start_idx = idx * self.batch_size
                end_idx = min((idx + 1) * self.batch_size, len(textBatch))
                batch = textBatch[start_idx:end_idx]
                curEmbeddings = self.embedder.embed_batch(batch)
                embeddings = np.vstack((embeddings, curEmbeddings)) if embeddings.size else curEmbeddings
        return embeddings

    def get_dim(self):
        return self.embedder.embedding_vector_size
