# memory_manager.py

import json
import os

from settings import (
    MAX_RECENT_EXCHANGES,
    MAX_SUMMARY_CHARS,
    MAX_USER_CHARS,
    SESSION_FILE
)


class ConversationMemory:

    def __init__(self):

        self.summary = ""

        self.recent = []

        self.archived_exchanges = 0

    def add_exchange(self, question, answer):
        """Store a successful question-and-answer exchange."""

        if not question.strip() or not answer.strip():
            raise ValueError(
                "Both messages must be nonempty."
            )

        self.recent.extend([
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ])

    def compress(self, service):
        """Summarize exchanges beyond the recent window."""

        compressed = 0

        while len(self.recent) > MAX_RECENT_EXCHANGES * 2:

            oldest = self.recent[:2]

            # Keep the original until summarization succeeds.
            new_summary = service.summarize(
                self.summary,
                oldest
            )

            self.summary = new_summary

            del self.recent[:2]

            self.archived_exchanges += 1

            compressed += 1

        return compressed

    def clear(self):
        """Clear active conversation memory."""

        self.summary = ""

        self.recent.clear()

        self.archived_exchanges = 0

    def save(self):
        """Save conversation state as JSON."""

        SESSION_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "version": 1,
            "summary": self.summary,
            "recent": self.recent,
            "archived_exchanges": self.archived_exchanges
        }

        tmp = SESSION_FILE.with_suffix(".tmp")

        tmp.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        os.replace(
            tmp,
            SESSION_FILE
        )

        return SESSION_FILE

    def load(self):
        """Load and validate a saved conversation."""

        if SESSION_FILE.stat().st_size > 1_000_000:
            raise ValueError(
                "Session file exceeds 1 MB."
            )

        data = json.loads(
            SESSION_FILE.read_text(
                encoding="utf-8"
            )
        )

        if (
            not isinstance(data, dict)
            or data.get("version") != 1
        ):
            raise ValueError(
                "Unsupported session format."
            )

        summary = data.get("summary")
        recent = data.get("recent")
        archived = data.get("archived_exchanges")

        if (
            not isinstance(summary, str)
            or len(summary) > MAX_SUMMARY_CHARS
        ):
            raise ValueError(
                "Invalid session summary."
            )

        if (
            not isinstance(recent, list)
            or len(recent) % 2
            or len(recent) > 100
        ):
            raise ValueError(
                "Invalid session messages."
            )

        if (
            type(archived) is not int
            or archived < 0
        ):
            raise ValueError(
                "Invalid archived-exchange count."
            )

        cleaned = []

        for index, item in enumerate(recent):

            expected_role = (
                "user"
                if index % 2 == 0
                else "assistant"
            )

            if (
                not isinstance(item, dict)
                or item.get("role") != expected_role
                or not isinstance(item.get("content"), str)
                or not item["content"].strip()
                or len(item["content"]) > 10000
            ):
                raise ValueError(
                    "Invalid saved message."
                )

            cleaned.append({
                "role": expected_role,
                "content": item["content"]
            })

        # Update the active state only after validation.
        self.summary = summary

        self.recent = cleaned

        self.archived_exchanges = archived

        return SESSION_FILE

    def delete_saved_file(self):
        """Delete the previously saved JSON file."""

        SESSION_FILE.unlink()

        return SESSION_FILE
