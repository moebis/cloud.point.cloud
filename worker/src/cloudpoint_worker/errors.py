"""Stable structured errors shared by worker subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkerFault(Exception):
    """An internal fault that can be transported without leaking an exception."""

    code: str
    message: str
    recoverable: bool
    details: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
