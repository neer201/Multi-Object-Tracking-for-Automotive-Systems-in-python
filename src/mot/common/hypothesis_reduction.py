from typing import List

import numpy as np

from .gaussian_density import GaussianDensity


class HypothesisReduction:
    @staticmethod
    def prune(hypotheses_weights: List[float], multi_hypotheses: List, threshold: float):
        """Prunes hypotheses with small weights

        Parameters
        ----------
        hypotheses_weights : List[float]
            the weights of different hypotheses in logarithmic scale
        multi_hypotheses : List
            describes hypotheses
        threshold : float
            hypotheses with weights smaller than this threshold will be discarded

        Returns
        -------
        new_hypotheses_weights : List[float]
            hypotheses weights after pruning in logarighmic scale
        new_multi_hypotheses : List
            hypotheses after pruning
        """
        selection_indices = np.array(hypotheses_weights) > threshold
        new_hypotheses_weights = np.array(hypotheses_weights)[selection_indices]
        new_multi_hypotheses = np.array(multi_hypotheses)[selection_indices]
        return new_hypotheses_weights.tolist(), new_multi_hypotheses.tolist()

    @staticmethod
    def cap(hypotheses_weights: List[float], multi_hypotheses: List, top_k: int):
        """keeps top_k hypotheses with the highest weights and discard the rest

        Parameters
        ----------
        hypotheses_weights : List[float]
            the weights of different hypotheses in logarithmic scale
        multi_hypotheses : List
            describes hypotheses
        top_k : int
            only keep top_k hypotheses

        Returns
        -------
        new_hypotheses_weights : List[float]
            hypotheses weights after capping in logarighmic scale
        new_multi_hypotheses : List
            hypotheses after capping
        """
        assert top_k >= 0, "only keep must be equal of larger than 0"
        selection_indices = np.argsort(hypotheses_weights)[:top_k]
        new_hypotheses_weights = np.array(hypotheses_weights)[selection_indices]
        new_multi_hypotheses = np.array(multi_hypotheses)[selection_indices]
        return new_hypotheses_weights.tolist(), new_multi_hypotheses.tolist()

    @staticmethod
    def merge(hypotheses_weights: List[float], multi_hypotheses: List, threshold: float):
        """Merges hypotheses with small Mahalanobis distance

        Parameters
        ----------
        hypotheses_weights : List[float]
            the weights of different hypotheses in logarithmic scale
        multi_hypotheses : List
            describes hypotheses
        threshold : float
            merging threshold
        density : Gaussian Density
            class

        Returns
        -------
        new_hypotheses_weights : List[float]
            hypotheses weights after merging in logarighmic scale
        new_multi_hypotheses : List
            hypotheses after merging
        """
        (
            new_hypotheses_weights,
            new_multi_hypotheses,
        ) = GaussianDensity.mixture_reduction(weights=hypotheses_weights, states=multi_hypotheses, threshold=threshold)
        return new_hypotheses_weights, new_multi_hypotheses
