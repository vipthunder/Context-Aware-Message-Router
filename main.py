from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from data_loader import ContextAggregator
from graph_router import LangGraphRouter


OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def process_one(message: dict, aggregator: ContextAggregator, router: LangGraphRouter) -> dict:
    context = aggregator.for_message(message)
    try:
        result = router.invoke(message, context)
    except Exception as exc:
        result = {
            "message_id": message.get("message_id", ""),
            "action": "digest",
            "message_type": "unknown",
            "reason": f"Fallback route after processing error: {type(exc).__name__}.",
            "confidence": 0.51,
            "evidence_message_ids": "none",
        }
    return {column: result.get(column, "none" if column == "evidence_message_ids" else "") for column in OUTPUT_COLUMNS}


def run(dataset_dir: Path, output_path: Path, workers: int) -> list[dict]:
    load_dotenv()
    messages = pd.read_csv(dataset_dir / "messages.csv", keep_default_na=False).to_dict("records")
    aggregator = ContextAggregator(dataset_dir)
    router = LangGraphRouter(dataset_dir)

    results: dict[str, dict] = {}
    max_workers = max(1, min(workers, len(messages) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, message, aggregator, router): message["message_id"] for message in messages}
        for future in as_completed(futures):
            message_id = futures[future]
            results[message_id] = future.result()

    ordered = [results[message["message_id"]] for message in messages]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    return ordered


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate WhatsApp notification router.")
    parser.add_argument("--dataset", default=str(root / "dataset"), help="Path to dataset directory.")
    parser.add_argument("--output", default=str(root / "output.csv"), help="Output CSV path.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker count.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = run(Path(args.dataset), Path(args.output), args.workers)
    print(f"Wrote {len(rows)} predictions to {args.output}")
