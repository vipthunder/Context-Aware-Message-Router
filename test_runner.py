from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import ContextAggregator
from main import OUTPUT_COLUMNS, process_one, run
from graph_router import LangGraphRouter
from media_processor import MediaProcessor


def test_output_contract() -> None:
    output_dir = Path(".test_output")
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "output.csv"
    rows = run(Path("dataset"), output, workers=2)
    written = pd.read_csv(output, keep_default_na=False)
    assert list(written.columns) == OUTPUT_COLUMNS
    assert len(rows) == len(pd.read_csv("dataset/messages.csv", keep_default_na=False))
    assert set(written["action"]).issubset({"notify", "digest", "mute"})
    assert set(written["message_type"]).issubset({"personal", "urgent", "event", "payment", "business_update", "promotion", "greeting", "forward", "spam", "scam", "unknown"})


def test_missing_context_does_not_crash() -> None:
    aggregator = ContextAggregator("dataset")
    router = LangGraphRouter("dataset")
    message = {
        "message_id": "hidden_missing_context",
        "user_id": "u_missing",
        "conversation_type": "group",
        "group_id": "group_missing",
        "business_id": "",
        "sender_user_id": "u_missing_sender",
        "created_at": "2026-08-02 12:00",
        "message_text": "Can someone share the meeting notes later? No rush.",
        "media_type": "",
        "media_id": "",
        "forwarded_count": "0",
    }
    result = process_one(message, aggregator, router)
    assert result["message_id"] == "hidden_missing_context"
    assert result["action"] in {"notify", "digest", "mute"}


def test_malformed_media_fallback() -> None:
    dataset = Path(".test_output")
    dataset.mkdir(exist_ok=True)
    bad = dataset / "bad.mp3"
    bad.write_bytes(b"")
    processor = MediaProcessor(dataset)
    msg = {"message_id": "bad_media", "media_type": "voice", "media_id": "vn_bad", "message_text": "text fallback"}
    enriched = processor.enrich(msg, {"voice_path": "bad.mp3"})
    assert enriched["media_text"] == ""
    assert enriched["message_text"] == "text fallback"


if __name__ == "__main__":
    test_output_contract()
    test_missing_context_does_not_crash()
    test_malformed_media_fallback()
    print("All tests passed.")
