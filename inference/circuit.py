from __future__ import annotations

import time
from dataclasses import dataclass


class CircuitOpenError(Exception):
    pass


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float | None = None
    half_open: bool = False
    trial_started_at: float | None = None


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int,
        recovery_seconds: float,
        trial_timeout: float | None = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        # A half-open trial request that exits without reporting success or
        # failure (client disconnect, admission rejection, abandonment) must
        # not wedge the circuit forever: after `trial_timeout` a fresh trial
        # is allowed.
        self.trial_timeout = trial_timeout
        self._state = CircuitState()

    def allow(self) -> None:
        state = self._state
        if state.opened_at is None:
            return
        elapsed = time.monotonic() - state.opened_at
        if elapsed < self.recovery_seconds:
            raise CircuitOpenError("backend circuit is open")
        if state.half_open:
            trial_expired = (
                self.trial_timeout is not None
                and state.trial_started_at is not None
                and time.monotonic() - state.trial_started_at > self.trial_timeout
            )
            if not trial_expired:
                # A single trial request is already probing the backend
                # (GW-12); concurrent callers must not flood it while it is
                # half-open.
                raise CircuitOpenError(
                    "backend circuit is open (half-open trial in progress)"
                )
        state.half_open = True
        state.trial_started_at = time.monotonic()

    def abandon_trial(self) -> None:
        """Release a half-open trial whose outcome was never reported.

        Called when an admitted trial request exits without reaching the
        backend (admission rejection, cancellation before the backend call):
        the circuit stays open, but the next caller may start a fresh trial
        instead of every caller being rejected forever.
        """
        if self._state.half_open:
            self._state.half_open = False
            self._state.trial_started_at = None

    def success(self) -> None:
        self._state = CircuitState()

    def failure(self) -> None:
        state = self._state
        if state.half_open:
            state.opened_at = time.monotonic()
            state.half_open = False
            state.trial_started_at = None
            return
        state.failures += 1
        if state.failures >= self.failure_threshold:
            state.opened_at = time.monotonic()

    def status(self) -> dict[str, object]:
        state = self._state
        return {
            "state": "open" if state.opened_at is not None else "closed",
            "failures": state.failures,
            "half_open": state.half_open,
            "opened_at": state.opened_at,
        }
