"""Wall-clock insurance for a single measurement.

Why this exists, and what it is *not*.

A benchmark cell runs for hours against a network it does not control, so each
task gets a hard deadline enforced by a signal.  The signal interrupts a
blocking read in the main thread even if a library-level socket timeout has
already been satisfied in theory, and the evaluator records the instance as
``aborted`` -- never as a failure, so a dead link cannot be mistaken for a
method that answered wrongly.

It is emphatically **not a way to speed anything up**, and the budget must be
derived from what a task actually costs rather than picked small.  Every
individual request already carries its own bound (``urllib`` gets
``timeout=25`` in :mod:`calix.websearch`, ``timeout=120`` in
:mod:`calix.llm`); what makes a task slow is the *number* of them.  A single
GAIA task makes on the order of 110 LLM calls -- six roles over ten rounds --
which at a realistic ~15-20 s per call is 25-35 minutes of legitimate
wall-clock time, before retries multiply it further.

That distinction had to be learned the hard way: a first version of this guard
used a 600 s budget, declared three calibration runs "hung I/O, most likely a
stalled HTTPS read", and discarded them.  Sampling the process showed nothing
of the kind -- it was doing ordinary work and simply had not finished.  The
diagnosis in the error message was more confident than the evidence, and it
cost a whole day.  Hence the budget below, and hence the wording of the error,
which now says what is known and nothing more.
"""
from __future__ import annotations

import os
import signal
import threading

# Deliberately generous: it is sized to the slowest legitimate task, not to a
# guess about when something "should" have been done.  Retries multiply it --
# ``CALIX_LLM_RETRIES=10`` can turn one call into ten -- so the ceiling has to
# leave room for a genuinely unlucky task rather than cutting off a normal one.
DEFAULT_BUDGET = 7200


class TaskTimeout(Exception):
    """A task exceeded its wall-clock budget."""


def budget(default: int = DEFAULT_BUDGET) -> int:
    """Per-task wall-clock budget in seconds (``CALIX_TASK_TIMEOUT``)."""
    try:
        value = int(os.environ.get("CALIX_TASK_TIMEOUT", "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


class task_deadline:
    """Interrupt the enclosed block if it runs past ``seconds`` wall clock.

    Uses ``SIGALRM`` (Unix, main thread only -- the evaluation loops are
    single-threaded).  Nested use is refused rather than silently losing the
    outer deadline.
    """

    def __init__(self, seconds: int | None = None, what: str = "task"):
        self.seconds = seconds or budget()
        self.what = what
        self._previous = None

    def __enter__(self) -> "task_deadline":
        # ``SIGALRM`` cannot be armed outside the main thread -- ``signal.signal``
        # raises there -- so the deadline silently becomes a no-op inside a
        # thread pool.  That is not an oversight but the only correct choice:
        # delivering an alarm to an arbitrary worker would raise somewhere
        # unrelated.  Every request still carries its own socket timeout, so a
        # worker task cannot block forever; it merely has no task-level ceiling.
        if threading.current_thread() is not threading.main_thread():
            self._previous = None
            return self
        if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
            raise RuntimeError(
                "a task deadline is already armed; nested deadlines are unsafe")
        self._previous = signal.signal(signal.SIGALRM, self._fire)
        signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
        return self

    def _fire(self, signum, frame):  # pragma: no cover - signal handler
        raise TaskTimeout(
            f"{self.what} exceeded its {self.seconds}s wall-clock budget "
            "(individual requests are bounded, so this is either an unusual "
            "number of retries or genuine pathological blocking)")

    def __exit__(self, *exc) -> bool:
        # Leave the timer armed for nobody; ``0`` must be passed before the
        # handler is restored, or a pending alarm fires into the old handler.
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        if self._previous is not None:
            signal.signal(signal.SIGALRM, self._previous)
        return False


class CircuitBreaker:
    """Stop spending a cell's budget once the infrastructure is plainly down.

    The 2026-10-29 outage was expensive not because tasks failed but because
    they kept being *attempted*: 229 of 350 tasks each burned their full retry
    budget against an endpoint that answered nothing, and the job still exited
    0 with TSR values in the log.  After enough consecutive tasks abort for
    infrastructure reasons, the rest are marked aborted straight away rather
    than re-discovering the same outage one half-hour task at a time.

    This only ever converts *attempted* work into *recorded absence* -- it can
    never turn a graded row into a failure, and it never makes a run look more
    successful than it was.
    """

    def __init__(self, threshold: int | None = None):
        if threshold is None:
            try:
                threshold = int(os.environ.get("CALIX_ABORT_THRESHOLD", "6"))
            except ValueError:
                threshold = 6
        self.threshold = threshold
        self.consecutive = 0
        self._tripped = False
        self._lock = threading.Lock()

    @property
    def tripped(self) -> bool:
        """True once the breaker has opened; stays open for the whole cell."""
        return self._tripped

    def record(self, aborted: bool) -> bool:
        """Record one task outcome; return True if the breaker is now open."""
        with self._lock:
            if aborted:
                self.consecutive += 1
                if self.threshold and self.consecutive >= self.threshold:
                    self._tripped = True
            else:
                self.consecutive = 0
            return self._tripped
