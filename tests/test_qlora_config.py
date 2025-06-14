"""Tests for QLoRA configuration module."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.platform_config import (  # noqa: E402  # type: ignore
    get_device_map,
    get_gradient_accumulation_steps,
    get_optimal_qwen_model,
    get_training_batch_size,
    is_cloud_environment,
)
from qlora_config import (  # noqa: E402  # type: ignore
    setup_qlora_config,
    setup_training_arguments,
)


class TestPlatformConfig:
    """Test platform configuration functions."""

    def test_get_optimal_qwen_model_local(self) -> None:
        """Test model selection for local environment."""
        with patch("config.platform_config.is_cloud_environment", return_value=False):
            model = get_optimal_qwen_model()
            assert model == "Qwen/Qwen2-0.5B"

    def test_get_optimal_qwen_model_cloud(self) -> None:
        """Test model selection for cloud environment."""
        with patch("config.platform_config.is_cloud_environment", return_value=True):
            model = get_optimal_qwen_model()
            assert model == "Qwen/Qwen2-1.5B"

    def test_get_optimal_qwen_model_override(self) -> None:
        """Test model selection with environment variable override."""
        with patch.dict("os.environ", {"QWEN_MODEL_SIZE": "1.5B"}):
            model = get_optimal_qwen_model()
            assert model == "Qwen/Qwen2-1.5B"

    def test_get_device_map_cuda(self) -> None:
        """Test device mapping with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            device_map = get_device_map()
            assert device_map == "auto"

    def test_get_device_map_mps(self) -> None:
        """Test device mapping with MPS available."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            device_map = get_device_map()
            assert device_map == "cpu"  # MPS has limitations

    def test_get_device_map_cpu(self) -> None:
        """Test device mapping with only CPU available."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            device_map = get_device_map()
            assert device_map == "cpu"

    def test_training_batch_size_cloud(self) -> None:
        """Test batch size for cloud environment."""
        with patch("config.platform_config.is_cloud_environment", return_value=True):
            batch_size = get_training_batch_size()
            assert batch_size == 4

    def test_training_batch_size_local(self) -> None:
        """Test batch size for local environment."""
        with patch("config.platform_config.is_cloud_environment", return_value=False):
            batch_size = get_training_batch_size()
            assert batch_size == 2

    def test_gradient_accumulation_cloud(self) -> None:
        """Test gradient accumulation for cloud environment."""
        with patch("config.platform_config.is_cloud_environment", return_value=True):
            steps = get_gradient_accumulation_steps()
            assert steps == 4

    def test_gradient_accumulation_local(self) -> None:
        """Test gradient accumulation for local environment."""
        with patch("config.platform_config.is_cloud_environment", return_value=False):
            steps = get_gradient_accumulation_steps()
            assert steps == 8


class TestQLoRAConfig:
    """Test QLoRA configuration functions."""

    def test_setup_qlora_config(self) -> None:
        """Test QLoRA configuration setup."""
        config = setup_qlora_config()

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"

        # Check target modules for Qwen2
        expected_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        # Check target modules for Qwen2 (can be list or set depending on PEFT version)
        assert config.target_modules is not None
        if isinstance(config.target_modules, list):
            assert set(config.target_modules) == set(expected_modules)
        else:
            # PEFT may convert to set internally
            assert config.target_modules == set(expected_modules)

    def test_setup_training_arguments(self) -> None:
        """Test training arguments setup."""
        # Test actual function behavior without mocking platform detection
        args = setup_training_arguments()

        # Batch size can be 2 (local) or 4 (cloud/docker) depending on environment
        assert args.per_device_train_batch_size in [2, 4]
        assert args.per_device_eval_batch_size in [2, 4]
        # Gradient accumulation can be 4 (cloud) or 8 (local) depending on environment
        assert args.gradient_accumulation_steps in [4, 8]
        assert args.num_train_epochs == 3
        assert args.fp16 is False  # No CUDA available
        assert args.bf16 is False  # Disabled for compatibility
        assert args.save_strategy == "epoch"
        assert args.eval_strategy == "epoch"

    def test_setup_training_arguments_mps(self) -> None:
        """Test training arguments setup for Apple Silicon."""
        # Test with mocked MPS availability and force local environment
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
            patch("config.platform_config.is_cloud_environment", return_value=False),
        ):
            args = setup_training_arguments()

            assert args.fp16 is False  # No CUDA
            assert args.bf16 is False  # Disabled for compatibility
            assert args.per_device_train_batch_size == 2  # Local environment
            assert args.gradient_accumulation_steps == 8  # Local environment


class TestCloudDetection:
    """Test cloud environment detection."""

    def test_is_cloud_environment_colab(self) -> None:
        """Test detection of Google Colab environment."""
        with patch.dict("os.environ", {"COLAB_GPU": "1"}):
            assert is_cloud_environment() is True

    def test_is_cloud_environment_aws(self) -> None:
        """Test detection of AWS environment."""
        with patch.dict("os.environ", {"AWS_EXECUTION_ENV": "AWS_ECS_FARGATE"}):
            assert is_cloud_environment() is True

    def test_is_cloud_environment_docker(self) -> None:
        """Test detection of Docker environment."""
        with patch("os.path.exists", return_value=True):
            assert is_cloud_environment() is True

    def test_is_cloud_environment_high_memory(self) -> None:
        """Test detection based on high memory."""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB

        with patch("psutil.virtual_memory", return_value=mock_memory):
            assert is_cloud_environment() is True

    def test_is_cloud_environment_local(self) -> None:
        """Test detection of local environment."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("os.path.exists", return_value=False),
            patch("psutil.virtual_memory") as mock_psutil,
        ):
            mock_memory = Mock()
            mock_memory.total = 8 * 1024**3  # 8GB
            mock_psutil.return_value = mock_memory

            assert is_cloud_environment() is False

    def test_is_cloud_environment_no_psutil(self) -> None:
        """Test detection when psutil is not available."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("os.path.exists", return_value=False),
            patch("psutil.virtual_memory", side_effect=ImportError),
        ):
            assert is_cloud_environment() is False
