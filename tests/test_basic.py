"""Basic tests to verify the testing setup."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_python_version() -> None:
    """Test that we're using the correct Python version."""
    assert sys.version_info >= (3, 11)


def test_imports() -> None:
    """Test that we can import our main dependencies."""
    try:
        import accelerate  # noqa: F401
        import datasets  # noqa: F401
        import gradio  # noqa: F401
        import pandas  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        assert False, f"Failed to import required dependency: {e}"


def test_src_package() -> None:
    """Test that we can import our src package."""
    try:
        import src  # noqa: F401

        assert hasattr(src, "__version__")
        assert src.__version__ == "0.1.0"
    except ImportError as e:
        assert False, f"Failed to import src package: {e}"
