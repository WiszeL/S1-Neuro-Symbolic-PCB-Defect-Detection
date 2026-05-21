from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


_PREDICTION_CHUNK_SIZE = 512


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
        original_input_dim: int | None = None,
        selected_feature_indices: np.ndarray | None = None,
        feature_shape: tuple[int, int, int] | None = None,
        class_names: tuple[str, ...] | None = None,
        leaf_smoothing: float = 1.0,
    ) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")

        self.max_depth = max_depth
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.original_input_dim = int(
            original_input_dim if original_input_dim is not None else input_dim
        )
        self.feature_shape = feature_shape
        self.class_names = class_names
        self.leaf_smoothing = float(leaf_smoothing)
        if selected_feature_indices is None:
            self.selected_feature_indices = None
        else:
            selected_indices = np.asarray(
                selected_feature_indices, dtype=np.int64
            ).reshape(-1)
            if selected_indices.shape[0] != input_dim:
                raise ValueError(
                    "selected_feature_indices must have one entry per retained symbolic feature."
                )
            self.selected_feature_indices = selected_indices

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
        # Preserve memmap arrays when dtype already matches to avoid
        # materialising the entire dataset (~3 GB) into RAM.
        if features.dtype == np.float32:
            prepared = features
        else:
            prepared = np.asarray(features, dtype=np.float32)
        if prepared.ndim == 1:
            prepared = prepared.reshape(1, -1)

        if prepared.shape[1] == self.input_dim:
            return prepared
        if (
            prepared.shape[1] == self.original_input_dim
            and self.selected_feature_indices is not None
        ):
            return prepared[:, self.selected_feature_indices]
        raise ValueError(
            "Feature dimensionality does not match the tree input. "
            f"Expected {self.input_dim} retained features or {self.original_input_dim} original features, "
            f"got {prepared.shape[1]}."
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

    def predict_log_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.clip(self.predict_proba(features), 1e-8, 1.0)
        return np.log(probabilities)

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

    def _to_original_feature_vector(self, retained_vector: np.ndarray) -> np.ndarray:
        retained = np.asarray(retained_vector, dtype=np.float32).reshape(-1)
        if retained.shape[0] == self.original_input_dim:
            return retained.copy()
        if retained.shape[0] != self.input_dim:
            raise ValueError(
                "Feature vector dimensionality does not match the tree input or original feature space."
            )
        if self.selected_feature_indices is None:
            return retained.copy()

        original = np.zeros((self.original_input_dim,), dtype=np.float32)
        original[self.selected_feature_indices] = retained
        return original

    def node_feature_indices(
        self,
        node_index: int,
        original_space: bool = False,
    ) -> np.ndarray:
        indices = np.flatnonzero(self.node_weights[node_index]).astype(np.int64)
        if not original_space or self.selected_feature_indices is None:
            return indices
        return self.selected_feature_indices[indices]

    def path_feature_indices(
        self,
        feature: np.ndarray,
        original_space: bool = False,
    ) -> np.ndarray:
        path = self.decision_path(feature)
        if not path:
            return np.zeros((0,), dtype=np.int64)

        per_node = [
            self.node_feature_indices(step.node_index, original_space=original_space)
            for step in path
        ]
        non_empty = [indices for indices in per_node if indices.size > 0]
        if not non_empty:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.concatenate(non_empty, axis=0)).astype(np.int64)

    def path_contribution_vector(
        self,
        feature: np.ndarray,
        original_space: bool = False,
        path: list[PathStep] | None = None,
    ) -> np.ndarray:
        prepared_feature = self._prepare_features(feature)[0]
        if path is None:
            path = self.decision_path(prepared_feature)
        if not path:
            output_dim = self.original_input_dim if original_space else self.input_dim
            return np.zeros((output_dim,), dtype=np.float32)

        contribution_vector = np.zeros((self.input_dim,), dtype=np.float32)
        for step in path:
            direction = 1.0 if step.went_left else -1.0
            contribution_vector += (
                direction * self.node_weights[step.node_index] * prepared_feature
            )

        if not original_space:
            return contribution_vector
        return self._to_original_feature_vector(contribution_vector)

    def summarize_path(
        self,
        feature: np.ndarray,
        path: list[PathStep] | None = None,
    ) -> dict[str, Any]:
        prepared_feature = self._prepare_features(feature)[0]
        if path is None:
            path = self.decision_path(prepared_feature)
        active_original_indices = self.path_feature_indices(
            prepared_feature, original_space=True
        )
        per_node_counts = [
            int(self.node_feature_indices(step.node_index, original_space=True).size)
            for step in path
        ]
        return {
            "path_length": len(path),
            "active_path_original_feature_count": active_original_indices.size,
            "mean_active_original_features_per_node": float(np.mean(per_node_counts))
            if per_node_counts
            else 0.0,
        }

    def node_weight_full(self, node_index: int) -> np.ndarray:
        if self.selected_feature_indices is None:
            return self.node_weights[node_index].copy()

        full_weights = np.zeros((self.original_input_dim,), dtype=np.float32)
        full_weights[self.selected_feature_indices] = self.node_weights[node_index]
        return full_weights

    def node_weight_grid(self, node_index: int) -> np.ndarray:
        if self.feature_shape is None:
            raise ValueError(
                "feature_shape is required to reshape node weights into a spatial grid."
            )
        return self.node_weight_full(node_index).reshape(self.feature_shape)

    def nonzero_weight_counts(self) -> list[int]:
        return np.count_nonzero(self.node_weights, axis=1).tolist()

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "original_input_dim": self.original_input_dim,
            "selected_feature_indices": (
                torch.from_numpy(self.selected_feature_indices.copy())
                if self.selected_feature_indices is not None
                else None
            ),
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
            original_input_dim=int(
                state_dict.get("original_input_dim", state_dict["input_dim"])
            ),
            selected_feature_indices=(
                state_dict["selected_feature_indices"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
                if state_dict.get("selected_feature_indices") is not None
                else None
            ),
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
