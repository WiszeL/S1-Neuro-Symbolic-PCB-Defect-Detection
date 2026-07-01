from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from util.features import ensure_float32

_PREDICTION_CHUNK_SIZE = 2048


@dataclass(frozen=True)
class PathStep:
    node_index: int
    score: float
    went_left: bool


class SparseObliqueDecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int,
        num_classes: int,
        input_dim: int,
        feature_shape: tuple[int, int, int] | None = None,
        class_names: tuple[str, ...] | None = None,
        leaf_smoothing: float = 1.0,
    ) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")

        self.max_depth = max_depth
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.feature_shape = feature_shape
        self.class_names = class_names
        self.leaf_smoothing = float(leaf_smoothing)

        self.num_internal_nodes = (2**max_depth) - 1
        self.num_leaves = 2**max_depth

        self.node_weights = np.zeros(
            (self.num_internal_nodes, input_dim), dtype=np.float32
        )
        self.node_bias = np.zeros((self.num_internal_nodes,), dtype=np.float32)
        self.leaf_labels = np.zeros((self.num_leaves,), dtype=np.int64)
        self.leaf_distributions = np.full(
            (self.num_leaves, num_classes),
            fill_value=1.0 / max(num_classes, 1),
            dtype=np.float32,
        )

    # ---------------------------------------------------------------------
    # Tree structure helpers
    # ---------------------------------------------------------------------

    def left_child(self, node_index: int) -> int:
        return (2 * node_index) + 1

    def right_child(self, node_index: int) -> int:
        return (2 * node_index) + 2

    def is_leaf_node(self, node_index: int) -> bool:
        return node_index >= self.num_internal_nodes

    def leaf_position(self, node_index: int) -> int:
        if not self.is_leaf_node(node_index):
            raise ValueError("leaf_position can only be called on a leaf node.")
        return node_index - self.num_internal_nodes

    # ---------------------------------------------------------------------
    # Prediction and path inspection
    # ---------------------------------------------------------------------

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        # ensure_float32 preserves memmap arrays when dtype already matches,
        # avoiding materialising the entire dataset (~3 GB) into RAM.
        prepared = ensure_float32(features)
        if prepared.ndim == 1:
            prepared = prepared.reshape(1, -1)

        if prepared.shape[1] == self.input_dim:
            return prepared
        raise ValueError(
            "Feature dimensionality does not match the tree input. "
            f"Expected {self.input_dim}, got {prepared.shape[1]}."
        )

    def _score_features_by_node_indices(
        self,
        features: np.ndarray,
        node_indices: np.ndarray,
    ) -> np.ndarray:
        weights_for_rows = self.node_weights[node_indices]
        biases_for_rows = self.node_bias[node_indices]
        return np.einsum("ij,ij->i", features, weights_for_rows) + biases_for_rows

    def predict_leaf_indices(self, features: np.ndarray) -> np.ndarray:
        features = self._prepare_features(features)
        leaf_indices = np.empty((features.shape[0],), dtype=np.int64)

        for start in range(0, features.shape[0], _PREDICTION_CHUNK_SIZE):
            stop = min(start + _PREDICTION_CHUNK_SIZE, features.shape[0])
            chunk_features = features[start:stop]
            node_indices = np.zeros((chunk_features.shape[0],), dtype=np.int64)

            for _ in range(self.max_depth):
                scores = self._score_features_by_node_indices(
                    chunk_features, node_indices
                )
                go_left = scores >= 0.0
                node_indices = np.where(
                    go_left, (2 * node_indices) + 1, (2 * node_indices) + 2
                )

            leaf_indices[start:stop] = node_indices - self.num_internal_nodes

        return leaf_indices

    def predict(self, features: np.ndarray) -> np.ndarray:
        leaf_indices = self.predict_leaf_indices(features)
        return self.leaf_labels[leaf_indices]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        leaf_indices = self.predict_leaf_indices(features)
        return self.leaf_distributions[leaf_indices]

    def predict_from_node(self, features: np.ndarray, node_index: int) -> np.ndarray:
        features = self._prepare_features(features)
        predictions = np.empty((features.shape[0],), dtype=np.int64)

        for start in range(0, features.shape[0], _PREDICTION_CHUNK_SIZE):
            stop = min(start + _PREDICTION_CHUNK_SIZE, features.shape[0])
            chunk_features = features[start:stop]
            current_nodes = np.full(
                (chunk_features.shape[0],), node_index, dtype=np.int64
            )

            while np.any(current_nodes < self.num_internal_nodes):
                internal_mask = current_nodes < self.num_internal_nodes
                active_positions = np.where(internal_mask)[0]
                active_nodes = current_nodes[active_positions]
                active_features = chunk_features[active_positions]
                scores = self._score_features_by_node_indices(
                    active_features, active_nodes
                )
                next_nodes = np.where(
                    scores >= 0.0, (2 * active_nodes) + 1, (2 * active_nodes) + 2
                )
                current_nodes[active_positions] = next_nodes

            predictions[start:stop] = self.leaf_labels[
                current_nodes - self.num_internal_nodes
            ]

        return predictions

    def decision_path(self, feature: np.ndarray) -> list[PathStep]:
        feature = self._prepare_features(feature)
        node_index = 0
        path: list[PathStep] = []

        while not self.is_leaf_node(node_index):
            score = float(
                (feature * self.node_weights[node_index]).sum()
                + self.node_bias[node_index]
            )
            went_left = score >= 0.0
            path.append(
                PathStep(node_index=node_index, score=score, went_left=went_left)
            )
            node_index = (
                self.left_child(node_index)
                if went_left
                else self.right_child(node_index)
            )

        return path

    def leaf_index_for_feature(self, feature: np.ndarray) -> int:
        feature = self._prepare_features(feature)
        return int(self.predict_leaf_indices(feature)[0])

    # ---------------------------------------------------------------------
    # Introspection and persistence
    # ---------------------------------------------------------------------

    def node_feature_indices(self, node_index: int) -> np.ndarray:
        return np.flatnonzero(self.node_weights[node_index]).astype(np.int64)

    def path_feature_indices(self, feature: np.ndarray) -> np.ndarray:
        path = self.decision_path(feature)
        if not path:
            return np.zeros((0,), dtype=np.int64)

        per_node = [self.node_feature_indices(step.node_index) for step in path]
        non_empty = [indices for indices in per_node if indices.size > 0]
        if not non_empty:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.concatenate(non_empty, axis=0)).astype(np.int64)

    def summarize_path(
        self,
        feature: np.ndarray,
        path: list[PathStep] | None = None,
    ) -> dict[str, Any]:
        prepared_feature = self._prepare_features(feature)[0]
        if path is None:
            path = self.decision_path(prepared_feature)
        active_indices = self.path_feature_indices(prepared_feature)
        per_node_counts = [
            int(self.node_feature_indices(step.node_index).size) for step in path
        ]
        return {
            "path_length": len(path),
            "active_path_original_feature_count": active_indices.size,
            "mean_active_original_features_per_node": float(np.mean(per_node_counts))
            if per_node_counts
            else 0.0,
        }

    def node_weight_grid(self, node_index: int) -> np.ndarray:
        if self.feature_shape is None:
            raise ValueError(
                "feature_shape is required to reshape node weights into a spatial grid."
            )
        return self.node_weights[node_index].copy().reshape(self.feature_shape)

    def nonzero_weight_counts(self) -> list[int]:
        return np.count_nonzero(self.node_weights, axis=1).tolist()

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "feature_shape": self.feature_shape,
            "class_names": self.class_names,
            "leaf_smoothing": self.leaf_smoothing,
            "node_weights": torch.from_numpy(self.node_weights.copy()),
            "node_bias": torch.from_numpy(self.node_bias.copy()),
            "leaf_labels": torch.from_numpy(self.leaf_labels.copy()),
            "leaf_distributions": torch.from_numpy(self.leaf_distributions.copy()),
        }

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any]
    ) -> "SparseObliqueDecisionTreeClassifier":
        tree = cls(
            max_depth=int(state_dict["max_depth"]),
            num_classes=int(state_dict["num_classes"]),
            input_dim=int(state_dict["input_dim"]),
            feature_shape=tuple(state_dict["feature_shape"])
            if state_dict["feature_shape"] is not None
            else None,
            class_names=tuple(state_dict["class_names"])
            if state_dict["class_names"] is not None
            else None,
            leaf_smoothing=float(state_dict.get("leaf_smoothing", 1.0)),
        )
        tree.node_weights = (
            state_dict["node_weights"].detach().cpu().numpy().astype(np.float32)
        )
        tree.node_bias = (
            state_dict["node_bias"].detach().cpu().numpy().astype(np.float32)
        )
        tree.leaf_labels = (
            state_dict["leaf_labels"].detach().cpu().numpy().astype(np.int64)
        )
        tree.leaf_distributions = (
            state_dict["leaf_distributions"].detach().cpu().numpy().astype(np.float32)
        )
        return tree
