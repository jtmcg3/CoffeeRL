#!/usr/bin/env python3
"""
Simple script to run the CoffeRL analytics dashboard.

Usage:
    python run_dashboard.py [--host HOST] [--port PORT] [--debug]
"""

import argparse
import os
import sys

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.dashboard import run_dashboard


def main():
    """Main entry point for dashboard runner."""
    parser = argparse.ArgumentParser(description="Run CoffeRL Analytics Dashboard")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to bind to (default: 8050)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print(f"Starting CoffeRL Analytics Dashboard...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"URL: http://{args.host}:{args.port}")
    print("\nPress Ctrl+C to stop the server")

    try:
        run_dashboard(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
