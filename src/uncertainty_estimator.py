"""Uncertainty estimation module using Monte Carlo dropout.

This module provides uncertainty estimation for PyTorch models using Monte Carlo dropout
during inference. It works with any model that has dropout layers and provides both
token-level and sequence-level uncertainty metrics.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class UncertaintyEstimator:
    """Estimates model uncertainty using Monte Carlo dropout.

    This class temporarily enables dropout during inference to estimate model
    uncertainty through multiple forward passes. It preserves the original
    model state and is compatible with any PyTorch model.
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_rate: float = 0.1,
        num_samples: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the uncertainty estimator.

        Args:
            model: PyTorch model to estimate uncertainty for
            dropout_rate: Dropout probability for Monte Carlo sampling
            num_samples: Number of forward passes for uncertainty estimation
            device: Device to run computations on (auto-detected if None)
        """
        self.model = model
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        self.device = device or next(model.parameters()).device

        # Original training state will be stored when needed
        self._original_training_state = None

        # Cache for dropout modules
        self._dropout_modules: List[nn.Module] = []
        self._original_dropout_states: List[bool] = []

        # Find and cache all dropout modules
        self._find_dropout_modules()

    def _find_dropout_modules(self) -> None:
        """Find all dropout modules in the model."""
        self._dropout_modules = []
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                self._dropout_modules.append(module)

    def _enable_dropout(self) -> None:
        """Enable dropout for all dropout modules and store original states."""
        self._original_dropout_states = []
        for module in self._dropout_modules:
            self._original_dropout_states.append(module.training)
            module.train()  # Enable dropout
            module.p = self.dropout_rate  # Set dropout rate

    def _restore_dropout_state(self) -> None:
        """Restore original dropout states."""
        for module, original_state in zip(
            self._dropout_modules, self._original_dropout_states
        ):
            module.train(original_state)

    def _restore_model_state(self) -> None:
        """Restore the model to its original training state."""
        if self._original_training_state is not None:
            self.model.train(self._original_training_state)
        self._restore_dropout_state()

    def estimate_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_predictions: bool = False,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """Estimate uncertainty using Monte Carlo dropout.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_predictions: Whether to return all predictions

        Returns:
            Dictionary containing uncertainty metrics:
            - mean_logits: Mean predictions across samples
            - variance: Variance of predictions
            - entropy: Predictive entropy
            - mutual_information: Mutual information (epistemic uncertainty)
            - total_uncertainty: Total uncertainty score
            - predictions: All predictions if return_predictions=True
        """
        # Ensure inputs are on correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Store original state and enable dropout
        self._original_training_state = self.model.training
        self._enable_dropout()

        try:
            # Collect predictions from multiple forward passes
            all_logits = []

            with torch.no_grad():
                for _ in range(self.num_samples):
                    # Forward pass with dropout enabled
                    if attention_mask is not None:
                        outputs = self.model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                    else:
                        outputs = self.model(input_ids=input_ids)

                    # Extract logits
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    else:
                        logits = outputs

                    all_logits.append(logits)

            # Stack all predictions [num_samples, batch_size, seq_len, vocab_size]
            all_logits = torch.stack(all_logits, dim=0)

            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_uncertainty_metrics(all_logits)

            if return_predictions:
                uncertainty_metrics["predictions"] = all_logits

            return uncertainty_metrics

        finally:
            # Always restore original model state
            self._restore_model_state()

    def _calculate_uncertainty_metrics(
        self, all_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculate various uncertainty metrics from multiple predictions.

        Args:
            all_logits: Logits from multiple forward passes [num_samples, batch_size, seq_len, vocab_size]

        Returns:
            Dictionary of uncertainty metrics
        """
        # Convert logits to probabilities
        all_probs = F.softmax(all_logits, dim=-1)

        # Mean predictions
        mean_probs = torch.mean(all_probs, dim=0)
        mean_logits = torch.mean(all_logits, dim=0)

        # Variance of predictions (aleatoric + epistemic uncertainty)
        variance = torch.var(all_probs, dim=0)

        # Predictive entropy (total uncertainty)
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-8), dim=-1
        )

        # Expected entropy (aleatoric uncertainty)
        individual_entropies = -torch.sum(
            all_probs * torch.log(all_probs + 1e-8), dim=-1
        )
        expected_entropy = torch.mean(individual_entropies, dim=0)

        # Mutual information (epistemic uncertainty)
        mutual_information = predictive_entropy - expected_entropy

        # Total uncertainty score (average across sequence)
        total_uncertainty = torch.mean(predictive_entropy, dim=-1)

        return {
            "mean_logits": mean_logits,
            "mean_probs": mean_probs,
            "variance": variance,
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_information,
            "total_uncertainty": total_uncertainty,
        }

    def get_uncertainty_threshold(
        self,
        validation_data: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
        percentile: float = 90.0,
    ) -> float:
        """Calculate uncertainty threshold based on validation data.

        Args:
            validation_data: List of (input_ids, attention_mask) tuples
            percentile: Percentile for threshold calculation

        Returns:
            Uncertainty threshold value
        """
        uncertainties = []

        for input_ids, attention_mask in validation_data:
            metrics = self.estimate_uncertainty(input_ids, attention_mask)
            uncertainty = metrics["total_uncertainty"].mean().item()
            uncertainties.append(uncertainty)

        return float(np.percentile(uncertainties, percentile))

    def should_generate_experiment(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Determine if an experiment should be generated based on uncertainty.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            threshold: Uncertainty threshold (uses default if None)

        Returns:
            Tuple of (should_generate, uncertainty_score)
        """
        metrics = self.estimate_uncertainty(input_ids, attention_mask)
        uncertainty_score = metrics["total_uncertainty"].mean().item()

        if threshold is None:
            # Use a default threshold (can be configured)
            threshold = 1.0  # Default threshold

        should_generate = uncertainty_score > threshold

        return should_generate, uncertainty_score


class UncertaintyCache:
    """Cache for storing dropout masks to improve performance."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of cached items
        """
        self.max_size = max_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.access_order: List[str] = []

    def get_key(self, shape: Tuple[int, ...], device: torch.device) -> str:
        """Generate cache key for given shape and device."""
        return f"{shape}_{device}"

    def get_mask(
        self, shape: Tuple[int, ...], device: torch.device, dropout_rate: float
    ) -> torch.Tensor:
        """Get or create dropout mask.

        Args:
            shape: Shape of the mask
            device: Device for the mask
            dropout_rate: Dropout probability

        Returns:
            Dropout mask tensor
        """
        key = self.get_key(shape, device)

        if key in self.cache:
            # Move to end of access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]

        # Create new mask
        mask = torch.bernoulli(torch.full(shape, 1 - dropout_rate, device=device))

        # Add to cache
        self._add_to_cache(key, mask)

        return mask

    def _add_to_cache(self, key: str, mask: torch.Tensor) -> None:
        """Add mask to cache with LRU eviction."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = mask
        self.access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
