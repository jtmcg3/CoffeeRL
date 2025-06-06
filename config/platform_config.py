"""Platform-specific configuration for CoffeeRL-Lite.

This module handles platform-specific dependencies and configurations,
particularly for packages like bitsandbytes that don't support all platforms.
"""

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


def print_platform_summary() -> None:
    """Print a summary of platform capabilities."""
    platform_info = get_platform_info()
    bnb_available = check_bitsandbytes_compatibility()

    print("=== CoffeeRL-Lite Platform Summary ===")
    print(f"System: {platform_info['system']} {platform_info['machine']}")
    print(f"Python: {platform_info['python_version'].split()[0]}")
    print(f"bitsandbytes support: {'✅' if bnb_available else '❌'}")

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
