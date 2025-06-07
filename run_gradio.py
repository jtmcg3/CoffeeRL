#!/usr/bin/env python3
"""Entry point for running the CoffeeRL-Lite Gradio web interface."""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from gradio_app import main  # noqa: E402

if __name__ == "__main__":
    main()
