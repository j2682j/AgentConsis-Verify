import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from .dense_embedding import DenseEmbedding


class BGEM3Embedding(DenseEmbedding):
    """BAAI/bge-m3 dense retrieval.

    Two differences from the E5 family matter for correctness:

    - Pooling is the CLS token, not a masked average.
    - No ``query:`` / ``passage:`` prefixes. Adding them measurably degrades
      bge-m3, so ``Embedder.prepare_*_text`` must leave its text alone.
    """

    def __init__(self, model_name_or_path: str = None):
        if model_name_or_path is None:
            model_name_or_path = "BAAI/bge-m3"
        super().__init__(
            model_name_or_path=model_name_or_path,
            embedding_vector_size=1024,
            pooling_type="cls",
        )

    def pooling(self, last_hidden_states, attention_mask) -> Tensor:
        del attention_mask
        return F.normalize(last_hidden_states[:, 0], p=2, dim=1)


__all__ = ["BGEM3Embedding"]
