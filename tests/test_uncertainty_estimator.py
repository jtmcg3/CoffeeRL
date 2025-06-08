"""Tests for uncertainty estimation module."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.uncertainty_estimator import UncertaintyCache, UncertaintyEstimator


class MockModel(nn.Module):
    """Mock model for testing uncertainty estimation."""

    def __init__(
        self, vocab_size: int = 1000, hidden_size: int = 64, seq_len: int = 10
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Add some layers with dropout
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        """Forward pass with dropout layers."""
        x = self.embedding(input_ids)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.dropout2(x)
        logits = self.linear2(x)

        # Return object with logits attribute (like transformers models)
        class Output:
            def __init__(self, logits):
                self.logits = logits

        return Output(logits)


class TestUncertaintyEstimator:
    """Test cases for UncertaintyEstimator class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return MockModel()

    @pytest.fixture
    def estimator(self, mock_model):
        """Create an uncertainty estimator for testing."""
        return UncertaintyEstimator(mock_model, dropout_rate=0.1, num_samples=5)

    @pytest.fixture
    def sample_input(self):
        """Create sample input data."""
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        return input_ids, attention_mask

    def test_initialization(self, mock_model):
        """Test uncertainty estimator initialization."""
        estimator = UncertaintyEstimator(mock_model, dropout_rate=0.2, num_samples=15)

        assert estimator.model is mock_model
        assert estimator.dropout_rate == 0.2
        assert estimator.num_samples == 15
        assert len(estimator._dropout_modules) == 2  # Two dropout layers in mock model

    def test_find_dropout_modules(self, estimator):
        """Test that dropout modules are correctly identified."""
        assert len(estimator._dropout_modules) == 2
        assert all(
            isinstance(module, nn.Dropout) for module in estimator._dropout_modules
        )

    def test_enable_and_restore_dropout(self, estimator):
        """Test dropout state management."""
        # Store original states
        original_states = [module.training for module in estimator._dropout_modules]
        # Store original rates for verification
        [module.p for module in estimator._dropout_modules]

        # Enable dropout
        estimator._enable_dropout()

        # Check that dropout is enabled and rate is set
        for module in estimator._dropout_modules:
            assert module.training is True
            assert module.p == 0.1

        # Restore dropout state
        estimator._restore_dropout_state()

        # Check that original states are restored
        for module, original_state in zip(estimator._dropout_modules, original_states):
            assert module.training == original_state

    def test_model_state_restoration(self, estimator):
        """Test that model training state is properly restored."""
        # Set model to eval mode
        estimator.model.eval()
        original_state = estimator.model.training

        # Enable dropout (should not change model training state)
        estimator._enable_dropout()

        # Restore model state
        estimator._restore_model_state()

        assert estimator.model.training == original_state

    def test_estimate_uncertainty_basic(self, estimator, sample_input):
        """Test basic uncertainty estimation functionality."""
        input_ids, attention_mask = sample_input

        metrics = estimator.estimate_uncertainty(input_ids, attention_mask)

        # Check that all expected metrics are present
        expected_keys = [
            "mean_logits",
            "mean_probs",
            "variance",
            "predictive_entropy",
            "expected_entropy",
            "mutual_information",
            "total_uncertainty",
        ]
        for key in expected_keys:
            assert key in metrics

        # Check tensor shapes
        batch_size, seq_len = input_ids.shape
        vocab_size = estimator.model.vocab_size

        assert metrics["mean_logits"].shape == (batch_size, seq_len, vocab_size)
        assert metrics["mean_probs"].shape == (batch_size, seq_len, vocab_size)
        assert metrics["variance"].shape == (batch_size, seq_len, vocab_size)
        assert metrics["total_uncertainty"].shape == (batch_size,)

    def test_estimate_uncertainty_without_attention_mask(self, estimator, sample_input):
        """Test uncertainty estimation without attention mask."""
        input_ids, _ = sample_input

        metrics = estimator.estimate_uncertainty(input_ids)

        # Should work without attention mask
        assert "total_uncertainty" in metrics
        assert metrics["total_uncertainty"].shape == (input_ids.shape[0],)

    def test_estimate_uncertainty_with_predictions(self, estimator, sample_input):
        """Test uncertainty estimation with prediction return."""
        input_ids, attention_mask = sample_input

        metrics = estimator.estimate_uncertainty(
            input_ids, attention_mask, return_predictions=True
        )

        # Check that predictions are included
        assert "predictions" in metrics

        # Check predictions shape [num_samples, batch_size, seq_len, vocab_size]
        expected_shape = (
            estimator.num_samples,
            input_ids.shape[0],
            input_ids.shape[1],
            estimator.model.vocab_size,
        )
        assert metrics["predictions"].shape == expected_shape

    def test_uncertainty_metrics_calculation(self, estimator):
        """Test uncertainty metrics calculation with known data."""
        # Create synthetic logits with known properties
        num_samples, batch_size, seq_len, vocab_size = 3, 1, 2, 4

        # Create logits where first position has high uncertainty, second has low
        all_logits = torch.zeros(num_samples, batch_size, seq_len, vocab_size)

        # High uncertainty position (varying predictions)
        all_logits[0, 0, 0, :] = torch.tensor([2.0, 1.0, 0.5, 0.1])
        all_logits[1, 0, 0, :] = torch.tensor([0.1, 2.0, 1.0, 0.5])
        all_logits[2, 0, 0, :] = torch.tensor([0.5, 0.1, 2.0, 1.0])

        # Low uncertainty position (consistent predictions)
        all_logits[:, 0, 1, :] = torch.tensor([3.0, 0.1, 0.1, 0.1])

        metrics = estimator._calculate_uncertainty_metrics(all_logits)

        # High uncertainty position should have higher entropy
        assert metrics["predictive_entropy"][0, 0] > metrics["predictive_entropy"][0, 1]

        # Mutual information should be positive (epistemic uncertainty)
        assert torch.all(metrics["mutual_information"] >= 0)

    def test_get_uncertainty_threshold(self, estimator):
        """Test uncertainty threshold calculation."""
        # Create validation data
        validation_data = []
        for _ in range(5):
            input_ids = torch.randint(0, 1000, (1, 8))
            attention_mask = torch.ones(1, 8)
            validation_data.append((input_ids, attention_mask))

        threshold = estimator.get_uncertainty_threshold(
            validation_data, percentile=80.0
        )

        assert isinstance(threshold, float)
        assert threshold > 0

    def test_should_generate_experiment(self, estimator, sample_input):
        """Test experiment generation decision."""
        input_ids, attention_mask = sample_input

        # Test with default threshold
        should_generate, uncertainty_score = estimator.should_generate_experiment(
            input_ids, attention_mask
        )

        assert isinstance(should_generate, bool)
        assert isinstance(uncertainty_score, float)
        assert uncertainty_score > 0

        # Test with custom threshold
        should_generate_low, _ = estimator.should_generate_experiment(
            input_ids, attention_mask, threshold=0.0
        )
        should_generate_high, _ = estimator.should_generate_experiment(
            input_ids, attention_mask, threshold=100.0
        )

        assert should_generate_low is True  # Low threshold should trigger
        assert should_generate_high is False  # High threshold should not trigger

    def test_device_handling(self, mock_model):
        """Test that device handling works correctly."""
        # Test with CPU
        estimator = UncertaintyEstimator(mock_model)
        assert estimator.device == torch.device("cpu")

        # Test device detection from model
        if torch.cuda.is_available():
            mock_model.cuda()
            estimator = UncertaintyEstimator(mock_model)
            assert estimator.device.type == "cuda"

    def test_model_state_preservation(self, estimator, sample_input):
        """Test that model state is preserved after uncertainty estimation."""
        input_ids, attention_mask = sample_input

        # Set model to eval mode
        estimator.model.eval()
        original_training_state = estimator.model.training

        # Run uncertainty estimation
        estimator.estimate_uncertainty(input_ids, attention_mask)

        # Check that model state is preserved
        assert estimator.model.training == original_training_state

    def test_error_handling(self, estimator):
        """Test error handling in uncertainty estimation."""
        # Test with invalid input
        with pytest.raises(Exception):
            estimator.estimate_uncertainty(torch.tensor([]))  # Empty tensor


class TestUncertaintyCache:
    """Test cases for UncertaintyCache class."""

    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        return UncertaintyCache(max_size=3)

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = UncertaintyCache(max_size=5)
        assert cache.max_size == 5
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0

    def test_get_key(self, cache):
        """Test cache key generation."""
        shape = (10, 20)
        device = torch.device("cpu")
        key = cache.get_key(shape, device)

        assert isinstance(key, str)
        assert str(shape) in key
        assert str(device) in key

    def test_get_mask_creation(self, cache):
        """Test mask creation and caching."""
        shape = (5, 10)
        device = torch.device("cpu")
        dropout_rate = 0.1

        # First call should create mask
        mask1 = cache.get_mask(shape, device, dropout_rate)
        assert mask1.shape == shape
        assert mask1.device == device
        assert len(cache.cache) == 1

        # Second call should return cached mask
        mask2 = cache.get_mask(shape, device, dropout_rate)
        assert torch.equal(mask1, mask2)
        assert len(cache.cache) == 1

    def test_cache_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        device = torch.device("cpu")
        dropout_rate = 0.1

        # Fill cache to capacity
        shapes = [(5, 5), (6, 6), (7, 7)]
        masks = []
        for shape in shapes:
            mask = cache.get_mask(shape, device, dropout_rate)
            masks.append(mask)

        assert len(cache.cache) == 3

        # Add one more item (should evict first)
        new_shape = (8, 8)
        cache.get_mask(new_shape, device, dropout_rate)

        assert len(cache.cache) == 3
        # First item should be evicted
        first_key = cache.get_key(shapes[0], device)
        assert first_key not in cache.cache

    def test_cache_access_order_update(self, cache):
        """Test that access order is updated correctly."""
        device = torch.device("cpu")
        dropout_rate = 0.1

        # Add items
        shapes = [(5, 5), (6, 6)]
        for shape in shapes:
            cache.get_mask(shape, device, dropout_rate)

        # Access first item again
        cache.get_mask(shapes[0], device, dropout_rate)

        # First item should be at end of access order
        first_key = cache.get_key(shapes[0], device)
        assert cache.access_order[-1] == first_key

    def test_cache_clear(self, cache):
        """Test cache clearing."""
        device = torch.device("cpu")
        dropout_rate = 0.1

        # Add some items
        cache.get_mask((5, 5), device, dropout_rate)
        cache.get_mask((6, 6), device, dropout_rate)

        assert len(cache.cache) > 0
        assert len(cache.access_order) > 0

        # Clear cache
        cache.clear()

        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0


class TestIntegration:
    """Integration tests for uncertainty estimation."""

    def test_end_to_end_uncertainty_estimation(self):
        """Test complete uncertainty estimation workflow."""
        # Create model and estimator
        model = MockModel(vocab_size=100, hidden_size=32)
        estimator = UncertaintyEstimator(model, dropout_rate=0.1, num_samples=3)

        # Create input data
        input_ids = torch.randint(0, 100, (2, 5))
        attention_mask = torch.ones(2, 5)

        # Run uncertainty estimation
        metrics = estimator.estimate_uncertainty(input_ids, attention_mask)

        # Verify results
        assert "total_uncertainty" in metrics
        assert metrics["total_uncertainty"].shape == (2,)
        assert torch.all(metrics["total_uncertainty"] > 0)

        # Test experiment generation decision
        should_generate, score = estimator.should_generate_experiment(
            input_ids, attention_mask, threshold=0.5
        )
        assert isinstance(should_generate, bool)
        assert score > 0

    def test_uncertainty_estimation_consistency(self):
        """Test that uncertainty estimation is reasonably consistent."""
        model = MockModel(vocab_size=50, hidden_size=16)
        estimator = UncertaintyEstimator(model, dropout_rate=0.1, num_samples=10)

        # Same input should give similar uncertainty scores
        input_ids = torch.randint(0, 50, (1, 3))

        scores = []
        for _ in range(3):
            metrics = estimator.estimate_uncertainty(input_ids)
            scores.append(metrics["total_uncertainty"].item())

        # Scores should be similar (within reasonable variance)
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Standard deviation should be much smaller than mean
        assert std_score < mean_score * 0.5  # Less than 50% of mean
