"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import yaml
from typing import List, Optional, Union

from loguru import logger


class Normalizer:
    def __init__(
        self,
        center_vector: Optional[Union[List[float], np.ndarray]] = None,
        l1: bool = False,
        l2: bool = False,
        center: bool = False,
    ):
        self.center_vector = (
            np.array(center_vector) if center_vector is not None else None
        )
        self.l1 = l1
        self.l2 = l2
        self.center_enabled = center

    def normalize_l1(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies L1 normalization to a batch of embeddings."""
        norms = np.linalg.norm(embeddings, ord=1, axis=1, keepdims=True)
        return np.divide(embeddings, norms, out=np.copy(embeddings), where=norms > 0)

    def normalize_l2(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies L2 normalization to a batch of embeddings."""
        norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        return np.divide(embeddings, norms, out=np.copy(embeddings), where=norms > 0)

    def center(self, embeddings: np.ndarray) -> np.ndarray:
        """Subtracts the center vector from the embeddings."""
        if self.center_vector is None:
            return embeddings

        if embeddings.shape[1] != len(self.center_vector):
            raise ValueError(
                f"Embedding size {embeddings.shape[1]} does not match center vector size {len(self.center_vector)}"
            )
        return embeddings - self.center_vector

    def transform(
        self, embeddings: Union[List[List[float]], np.ndarray]
    ) -> List[List[float]]:
        """Applies centering and normalization in sequence."""
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            is_single = True
        else:
            is_single = False

        result = embeddings
        if self.center_enabled:
            result = self.center(result)
        if self.l1:
            result = self.normalize_l1(result)
        if self.l2:
            result = self.normalize_l2(result)

        list_result = result.tolist()
        return list_result if not is_single else list_result[0]

    def to_yaml(self) -> str:
        """Returns the normalizer parameters as a YAML string."""
        data = {
            "l1": self.l1,
            "l2": self.l2,
            "center": self.center_enabled,
            "center_vector": self.center_vector.tolist()
            if self.center_vector is not None
            else None,
        }
        return yaml.dump(data)

    def save_to_yaml(self, path: str):
        """Saves the normalizer parameters to a YAML file."""
        with open(path, "w") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Normalizer":
        """Loads a normalizer from a YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls(
            center_vector=data.get("center_vector"),
            l1=data.get("l1", False),
            l2=data.get("l2", False),
            center=data.get("center", False),
        )

    @classmethod
    def load_from_yaml(cls, path: str) -> "Normalizer":
        """Loads a normalizer from a YAML file."""
        with open(path, "r") as f:
            return cls.from_yaml(f.read())


_default_normalizer: Optional["Normalizer"] = None


def set_default_normalizer(normalizer: Optional["Normalizer"]) -> None:
    """Install the normalizer every embedding client uses unless given one.

    ``None`` restores the built-in default, an L2 normalizer. The library
    never reads a normalizer from the environment: the agent server applies
    ``KAVALAI_EMBEDDING_NORMALIZER_YAML`` by calling this at start-up.
    """
    global _default_normalizer
    _default_normalizer = normalizer


def get_default_normalizer() -> "Normalizer":
    """The default normalizer — the one installed with
    :func:`set_default_normalizer`, or an L2 normalizer."""
    global _default_normalizer
    if _default_normalizer is None:
        logger.debug("No default normalizer defined, using L2 norm")
        _default_normalizer = Normalizer(l2=True)
    return _default_normalizer
