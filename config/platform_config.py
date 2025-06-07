"""Platform-specific configuration for CoffeeRL-Lite.

This module handles platform-specific dependencies and configurations,
particularly for packages like bitsandbytes that don't support all platforms.
Includes QLoRA-specific configurations for Qwen2 models.
"""

import os
import platform
import sys
import warnings
from typing import Any, Optional


def get_platform_info() -> dict[str, Any]:
    """Get current platform information."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": sys.version,
        "is_macos": platform.system() == "Darwin",
        "is_linux": platform.system() == "Linux",
        "is_windows": platform.system() == "Windows",
        "is_arm64": platform.machine() in ("arm64", "aarch64"),
    }


def check_bitsandbytes_compatibility() -> bool:
    """Check if bitsandbytes is compatible with current platform."""
    platform_info = get_platform_info()

    # bitsandbytes only supports Linux and Windows
    if platform_info["is_macos"]:
        return False

    return True


def import_bitsandbytes_safe() -> Optional[Any]:
    """Safely import bitsandbytes with fallback for unsupported platforms."""
    if not check_bitsandbytes_compatibility():
        warnings.warn(
            "bitsandbytes is not supported on this platform (macOS). "
            "Quantization features will be disabled. "
            "For quantization support, use Linux or Windows, or consider cloud deployment.",
            UserWarning,
            stacklevel=2,
        )
        return None

    try:
        import bitsandbytes as bnb

        return bnb
    except ImportError as e:
        warnings.warn(
            f"Failed to import bitsandbytes: {e}. "
            "Quantization features will be disabled.",
            UserWarning,
            stacklevel=2,
        )
        return None


def get_quantization_config(
    use_4bit: bool = True, use_8bit: bool = False
) -> Optional[dict[str, Any]]:
    """Get quantization configuration if bitsandbytes is available."""
    bnb = import_bitsandbytes_safe()

    if bnb is None:
        return None

    if use_4bit:
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    elif use_8bit:
        return {
            "load_in_8bit": True,
        }

    return None


def get_torch_dtype() -> Any:
    """Get appropriate torch dtype based on platform."""
    import torch

    platform_info = get_platform_info()

    # Use float16 on CUDA, bfloat16 on newer CPUs, float32 as fallback
    if torch.cuda.is_available():
        return torch.float16
    elif platform_info["is_macos"] and platform_info["is_arm64"]:
        # Apple Silicon supports bfloat16
        return torch.bfloat16
    else:
        return torch.float32


def is_cloud_environment() -> bool:
    """Detect if running in a cloud environment (Colab, AWS, etc.)."""
    # Check for common cloud environment indicators
    cloud_indicators = [
        "COLAB_GPU",  # Google Colab
        "AWS_EXECUTION_ENV",  # AWS Lambda/EC2
        "AZURE_CLIENT_ID",  # Azure
        "KAGGLE_KERNEL_RUN_TYPE",  # Kaggle
        "PAPERSPACE_NOTEBOOK_REPO_ID",  # Paperspace
    ]

    for indicator in cloud_indicators:
        if os.getenv(indicator):
            return True

    # Check for Docker environment
    if os.path.exists("/.dockerenv"):
        return True

    # Check for high-memory systems (likely cloud)
    try:
        import psutil

        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        if total_memory_gb > 12:  # Assume cloud if >12GB RAM
            return True
    except ImportError:
        pass

    return False


def get_optimal_qwen_model() -> str:
    """Select optimal Qwen2 model based on platform and resources."""
    # Allow manual override via environment variable
    manual_model = os.getenv("QWEN_MODEL_SIZE")
    if manual_model:
        if manual_model in ["0.5B", "1.5B"]:
            return f"Qwen/Qwen2-{manual_model}"
        else:
            warnings.warn(
                f"Invalid QWEN_MODEL_SIZE: {manual_model}. Using auto-detection.",
                UserWarning,
            )

    # Auto-detect based on environment
    if is_cloud_environment():
        return "Qwen/Qwen2-1.5B"
    else:
        return "Qwen/Qwen2-0.5B"


def get_device_map() -> str:
    """Get appropriate device mapping strategy."""
    import torch

    if torch.cuda.is_available():
        return "auto"
    elif torch.backends.mps.is_available():
        # For Apple Silicon, use CPU for now as MPS has limitations with some operations
        return "cpu"
    else:
        return "cpu"


def get_training_batch_size() -> int:
    """Get optimal batch size based on platform."""
    platform_info = get_platform_info()

    if is_cloud_environment():
        return 4  # Higher batch size for cloud with more memory
    elif platform_info["is_macos"] and platform_info["is_arm64"]:
        return 2  # Conservative for Apple Silicon
    else:
        return 2  # Conservative default


def get_gradient_accumulation_steps() -> int:
    """Get gradient accumulation steps based on platform."""
    if is_cloud_environment():
        return 4
    else:
        return 8  # More accumulation for smaller batch sizes


def print_platform_summary() -> None:
    """Print a summary of platform capabilities."""
    platform_info = get_platform_info()
    bnb_available = check_bitsandbytes_compatibility()

    print("=== CoffeeRL-Lite Platform Summary ===")
    print(f"System: {platform_info['system']} {platform_info['machine']}")
    print(f"Python: {platform_info['python_version'].split()[0]}")
    print(f"Cloud environment: {'✅' if is_cloud_environment() else '❌'}")
    print(f"bitsandbytes support: {'✅' if bnb_available else '❌'}")
    print(f"Optimal Qwen2 model: {get_optimal_qwen_model()}")
    print(f"Device mapping: {get_device_map()}")
    print(f"Training batch size: {get_training_batch_size()}")
    print(f"Gradient accumulation: {get_gradient_accumulation_steps()}")

    if not bnb_available:
        print("📝 Note: Quantization features disabled on this platform")
        print("   For quantization support, consider:")
        print("   - Using Linux or Windows")
        print("   - Cloud deployment (Google Colab, AWS, etc.)")
        print("   - Docker with Linux container")

    import torch

    print(f"PyTorch CUDA available: {'✅' if torch.cuda.is_available() else '❌'}")
    print(
        f"PyTorch MPS available: {'✅' if torch.backends.mps.is_available() else '❌'}"
    )
    print("=" * 40)


if __name__ == "__main__":
    print_platform_summary()
