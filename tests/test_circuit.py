import time

import pytest

from inference.circuit import CircuitBreaker, CircuitOpenError


def test_circuit_opens_and_recovers(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    circuit = CircuitBreaker(2, 10)

    circuit.allow()
    circuit.failure()
    circuit.allow()
    circuit.failure()

    with pytest.raises(CircuitOpenError):
        circuit.allow()

    now[0] += 11
    circuit.allow()
    assert circuit.status()["half_open"] is True
    circuit.success()
    assert circuit.status()["state"] == "closed"


def test_half_open_admits_only_one_probe(monkeypatch) -> None:
    """GW-12: concurrent callers must not flood the backend while it recovers."""
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    circuit = CircuitBreaker(1, 10)

    circuit.failure()  # opens the circuit

    now[0] += 11  # recovery window elapsed
    circuit.allow()  # first caller becomes the trial probe
    with pytest.raises(CircuitOpenError):
        circuit.allow()  # concurrent caller is rejected
    with pytest.raises(CircuitOpenError):
        circuit.allow()
    assert circuit.status()["half_open"] is True

    circuit.success()  # trial succeeded -> circuit closes
    assert circuit.status()["state"] == "closed"
    circuit.allow()  # callers admitted again


def test_half_open_failure_reopens_and_rearms(monkeypatch) -> None:
    """GW-12: a failed trial reopens the circuit and admits a new probe later."""
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    circuit = CircuitBreaker(1, 10)

    circuit.failure()  # opens the circuit
    now[0] += 11
    circuit.allow()  # trial probe admitted
    circuit.failure()  # trial failed -> circuit reopens

    with pytest.raises(CircuitOpenError):
        circuit.allow()

    now[0] += 11
    circuit.allow()  # a fresh trial is admitted after the next recovery window
    assert circuit.status()["half_open"] is True


def test_trial_timeout_allows_a_fresh_trial(monkeypatch) -> None:
    """GW-15: a half-open trial that never reports a verdict must not wedge."""
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    circuit = CircuitBreaker(1, 10, trial_timeout=5)

    circuit.failure()  # opens the circuit
    now[0] += 11
    circuit.allow()  # starts the trial
    now[0] += 6  # the trial never reported and exceeded its timeout
    circuit.allow()  # a fresh trial must be allowed instead of rejecting forever

    assert circuit.status()["half_open"] is True


def test_abandon_trial_releases_the_half_open_probe(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    circuit = CircuitBreaker(1, 10, trial_timeout=5)
    circuit.failure()
    now[0] += 11
    circuit.allow()

    circuit.abandon_trial()
    assert circuit.status()["half_open"] is False
    # A fresh trial can start immediately after abandonment.
    circuit.allow()
    assert circuit.status()["half_open"] is True


def test_scheduler_wires_the_trial_timeout() -> None:
    """The 0.5.x circuit fix is inert unless the scheduler passes trial_timeout."""
    from inference.config import settings
    from inference.scheduler import Scheduler

    instance = Scheduler()
    for breaker in instance._circuits.values():
        assert breaker.trial_timeout == settings.circuit_trial_timeout_seconds
