"""Module for command-line interface utilities."""

import argparse


def build_parser(description: str) -> argparse.ArgumentParser:
    """Build and return an argument parser with standard configuration options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the current date as the exclusive end date instead of the fixed research date.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where plots and JSON summaries are saved.",
    )
    return parser
