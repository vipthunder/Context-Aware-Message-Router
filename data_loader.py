from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def number(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value) or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ContextBundle:
    users: dict[str, dict[str, Any]]
    groups: dict[str, dict[str, Any]]
    group_members: dict[tuple[str, str], dict[str, Any]]
    businesses: dict[str, dict[str, Any]]
    user_business: dict[tuple[str, str], dict[str, Any]]
    history_by_user: dict[str, list[dict[str, Any]]]
    history_by_sender: dict[tuple[str, str], list[dict[str, Any]]]
    events_by_message: dict[str, dict[str, Any]]
    images: dict[str, str]
    voice_notes: dict[str, str]
    notification_summary: dict[str, list[dict[str, Any]]]


class ContextAggregator:
    """Loads dataset CSVs once and exposes O(1) lookup maps."""

    def __init__(self, dataset_dir: str | Path = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.context = self._load()

    def _read(self, filename: str) -> pd.DataFrame:
        path = self.dataset_dir / filename
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, keep_default_na=False)

    def _records_by(self, filename: str, key: str) -> dict[str, dict[str, Any]]:
        frame = self._read(filename)
        if frame.empty or key not in frame.columns:
            return {}
        return {clean(row[key]): row.to_dict() for _, row in frame.iterrows()}

    def _load(self) -> ContextBundle:
        users = self._records_by("users.csv", "user_id")
        groups = self._records_by("groups.csv", "group_id")
        businesses = self._records_by("business_accounts.csv", "business_id")

        group_members: dict[tuple[str, str], dict[str, Any]] = {}
        for _, row in self._read("group_members.csv").iterrows():
            group_members[(clean(row.get("group_id")), clean(row.get("user_id")))] = row.to_dict()

        user_business: dict[tuple[str, str], dict[str, Any]] = {}
        for _, row in self._read("user_business_history.csv").iterrows():
            user_business[(clean(row.get("user_id")), clean(row.get("business_id")))] = row.to_dict()

        events_by_message = self._records_by("message_events.csv", "message_id")
        history_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        history_by_sender: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for _, row in self._read("message_history.csv").iterrows():
            record = row.to_dict()
            user_id = clean(record.get("user_id"))
            sender = clean(record.get("sender_user_id") or record.get("business_id"))
            history_by_user[user_id].append(record)
            if sender:
                history_by_sender[(user_id, sender)].append(record)

        images = {clean(r.get("image_id")): clean(r.get("file_path")) for _, r in self._read("images.csv").iterrows()}
        voice_notes = {clean(r.get("voice_note_id")): clean(r.get("file_path")) for _, r in self._read("voice_notes.csv").iterrows()}

        notification_summary: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for _, row in self._read("daily_notification_summary.csv").iterrows():
            notification_summary[clean(row.get("user_id"))].append(row.to_dict())

        return ContextBundle(
            users=users,
            groups=groups,
            group_members=group_members,
            businesses=businesses,
            user_business=user_business,
            history_by_user=dict(history_by_user),
            history_by_sender=dict(history_by_sender),
            events_by_message=events_by_message,
            images=images,
            voice_notes=voice_notes,
            notification_summary=dict(notification_summary),
        )

    def for_message(self, message: dict[str, Any]) -> dict[str, Any]:
        user_id = clean(message.get("user_id"))
        group_id = clean(message.get("group_id"))
        business_id = clean(message.get("business_id"))
        sender = clean(message.get("sender_user_id") or business_id)
        business = dict(self.context.businesses.get(business_id, {}))
        messages_sent = max(number(business.get("messages_sent_30d")), 1.0)
        business["global_mute_rate"] = min(number(business.get("user_reports_30d")) / messages_sent, 1.0)

        return {
            "user": self.context.users.get(user_id, {}),
            "group": self.context.groups.get(group_id, {}),
            "group_member": self.context.group_members.get((group_id, user_id), {}),
            "business": business,
            "user_business": self.context.user_business.get((user_id, business_id), {}),
            "history_user": self.context.history_by_user.get(user_id, []),
            "history_sender": self.context.history_by_sender.get((user_id, sender), []),
            "events_by_message": self.context.events_by_message,
            "image_path": self.context.images.get(clean(message.get("media_id")), ""),
            "voice_path": self.context.voice_notes.get(clean(message.get("media_id")), ""),
            "notification_days": self.context.notification_summary.get(user_id, []),
        }
