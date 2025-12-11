#!/usr/bin/env python3
"""Run persona replay learning diagnostics for a product."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.utils.inference.belief_manager.belief_manager import (
    run_replay_learning_job,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay persona journeys and emit diagnostics")
    parser.add_argument("product_id", help="Product ID to process")
    parser.add_argument(
        "--accounts",
        nargs="*",
        help="Optional list of account IDs. If omitted, all target accounts are scanned.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of accounts to process.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON summary. If omitted the summary is printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_replay_learning_job(
        args.product_id,
        account_ids=args.accounts,
        limit=args.limit,
        output_path=args.output,
    )
    if not args.output:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
