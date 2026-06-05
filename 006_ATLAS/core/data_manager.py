"""
core/data_manager.py
--------------------
Centralized data layer for ATLAS.
Loads and filters profile, calendar, and task JSON data for all agents.
"""

import json
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta


class DataManager:
    """
    Unified interface for all structured data sources.

    Supported sources:
        - Student profile  (profile.json)
        - Calendar events  (calendar_events.json)
        - Active tasks     (task.json)

    All data starts as None until loaded via load_data().
    """

    def __init__(self):
        self.profile_data = None
        self.calendar_data = None
        self.task_data = None

    # ── Loaders ───────────────────────────────────────────────────────────────
    def load_data(self, profile_json: str, calendar_json: str, task_json: str) -> None:
        """
        Parse and store all three JSON data sources.

        Args:
            profile_json:  Raw JSON string of the student profile.
            calendar_json: Raw JSON string of calendar events.
            task_json:     Raw JSON string of tasks / assignments.
        """
        self.profile_data = json.loads(profile_json)
        self.calendar_data = json.loads(calendar_json)
        self.task_data = json.loads(task_json)

    # ── Profile ───────────────────────────────────────────────────────────────
    def get_student_profile(self, student_id: str) -> Optional[Dict]:
        """
        Return a student's profile dict by ID.

        Args:
            student_id: Unique student identifier (e.g. "student_123").

        Returns:
            Profile dict if found, otherwise None.
        """
        if self.profile_data:
            return next(
                (p for p in self.profile_data["profiles"] if p["id"] == student_id),
                None,
            )
        return None

    # ── Datetime helper ───────────────────────────────────────────────────────
    def parse_datetime(self, dt_str: str) -> datetime:
        """
        Parse an ISO-format datetime string and normalise to UTC.

        Handles both timezone-aware and timezone-naive strings.
        """
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc)
        except ValueError:
            dt = datetime.fromisoformat(dt_str)
            return dt.replace(tzinfo=timezone.utc)

    # ── Calendar ──────────────────────────────────────────────────────────────
    def get_upcoming_events(self, days: int = 7) -> List[Dict]:
        """
        Return calendar events that start within the next `days` days.

        Args:
            days: Look-ahead window in days (default 7).

        Returns:
            Chronologically ordered list of upcoming event dicts.
        """
        if not self.calendar_data:
            return []

        now = datetime.now(timezone.utc)
        future = now + timedelta(days=days)
        events = []

        for event in self.calendar_data.get("events", []):
            try:
                start_time = self.parse_datetime(event["start"]["dateTime"])
                if now <= start_time <= future:
                    events.append(event)
            except (KeyError, ValueError) as exc:
                print(f"⚠️  Skipping malformed event: {exc}")

        return events

    # ── Tasks ─────────────────────────────────────────────────────────────────
    def get_active_tasks(self) -> List[Dict]:
        """
        Return tasks that are not yet completed and are still due in the future.

        Each returned task is enriched with a `due_datetime` key.

        Returns:
            List of active task dicts.
        """
        if not self.task_data:
            return []

        now = datetime.now(timezone.utc)
        active_tasks = []

        for task in self.task_data.get("tasks", []):
            try:
                due_date = self.parse_datetime(task["due"])
                if task["status"] == "needsAction" and due_date > now:
                    task["due_datetime"] = due_date
                    active_tasks.append(task)
            except (KeyError, ValueError) as exc:
                print(f"⚠️  Skipping malformed task: {exc}")

        return active_tasks
