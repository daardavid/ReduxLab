"""
Analysis pipeline with explicit states for each analysis flow.

States: IDLE -> LOADING -> VALIDATING -> PREPROCESSING -> ANALYZING -> VISUALIZING -> DONE
                                                                                   -> ERROR

The UI subscribes to state change callbacks to update progress bars and status labels.
"""

import logging
import time
from enum import Enum, auto
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    IDLE = auto()
    LOADING = auto()
    VALIDATING = auto()
    PREPROCESSING = auto()
    ANALYZING = auto()
    VISUALIZING = auto()
    DONE = auto()
    ERROR = auto()


# ---------------------------------------------------------------------------
# Legal transition map
# ---------------------------------------------------------------------------
# Each key maps to the set of states it is allowed to transition *to*.
# Forward progression is enforced; any active state may transition to ERROR,
# and both terminal states (DONE / ERROR) may reset back to IDLE.
# ---------------------------------------------------------------------------
_LEGAL_TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.IDLE:          {PipelineState.LOADING, PipelineState.ERROR},
    PipelineState.LOADING:       {PipelineState.VALIDATING, PipelineState.ERROR},
    PipelineState.VALIDATING:    {PipelineState.PREPROCESSING, PipelineState.ERROR},
    PipelineState.PREPROCESSING: {PipelineState.ANALYZING, PipelineState.ERROR},
    PipelineState.ANALYZING:     {PipelineState.VISUALIZING, PipelineState.ERROR},
    PipelineState.VISUALIZING:   {PipelineState.DONE, PipelineState.ERROR},
    PipelineState.DONE:          {PipelineState.IDLE},
    PipelineState.ERROR:         {PipelineState.IDLE},
}

# Callback type aliases for readability
StateChangeCallback = Callable[[PipelineState, PipelineState, dict[str, Any]], None]
ProgressCallback = Callable[[str, int, int], None]


class AnalysisPipeline:
    """Orchestrates an analysis through well-defined states.

    Usage::

        pipeline = AnalysisPipeline("series")
        pipeline.on_state_change(my_callback)   # subscribe
        pipeline.on_progress(my_progress_cb)    # subscribe to progress

        pipeline.transition(PipelineState.LOADING)
        data = load_data(...)

        pipeline.transition(PipelineState.VALIDATING)
        validate(data)

        ...
        pipeline.transition(PipelineState.DONE, results=results)

    Parameters
    ----------
    analysis_type : str
        A human-readable label for the kind of analysis being run
        (e.g. ``"series"``, ``"cross"``, ``"panel"``).
    """

    def __init__(self, analysis_type: str) -> None:
        self._analysis_type: str = analysis_type
        self._state: PipelineState = PipelineState.IDLE
        self._context: dict[str, Any] = {}
        self._state_callbacks: list[StateChangeCallback] = []
        self._progress_callbacks: list[ProgressCallback] = []
        self._start_time: float = time.monotonic()

        logger.debug(
            "Pipeline created for '%s' analysis (state=%s)",
            self._analysis_type,
            self._state.name,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> PipelineState:
        """Return the current pipeline state."""
        return self._state

    @property
    def analysis_type(self) -> str:
        """Return the analysis type label."""
        return self._analysis_type

    @property
    def context(self) -> dict[str, Any]:
        """Return the accumulated context dictionary.

        The context holds any keyword data passed through
        :meth:`transition` or :meth:`fail` calls.
        """
        return self._context

    @property
    def elapsed(self) -> float:
        """Return seconds elapsed since the pipeline was created."""
        return time.monotonic() - self._start_time

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------

    def on_state_change(self, callback: StateChangeCallback) -> None:
        """Register a callback invoked on every state transition.

        Parameters
        ----------
        callback : callable(old_state, new_state, context)
            Called **after** the state has changed.  ``context`` is the
            pipeline's accumulated context dict at that point.

        Raises
        ------
        TypeError
            If *callback* is not callable.
        """
        if not callable(callback):
            raise TypeError(f"Expected a callable, got {type(callback).__name__}")
        self._state_callbacks.append(callback)

    def on_progress(self, callback: ProgressCallback) -> None:
        """Register a callback invoked when progress is reported.

        Parameters
        ----------
        callback : callable(step_name, step_index, total_steps)
            ``step_index`` is 0-based; ``total_steps`` is the expected
            total number of steps in the current phase.

        Raises
        ------
        TypeError
            If *callback* is not callable.
        """
        if not callable(callback):
            raise TypeError(f"Expected a callable, got {type(callback).__name__}")
        self._progress_callbacks.append(callback)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def transition(self, new_state: PipelineState, **context: Any) -> None:
        """Move the pipeline to *new_state*.

        Parameters
        ----------
        new_state : PipelineState
            The target state.
        **context
            Arbitrary keyword data merged into :attr:`context`.

        Raises
        ------
        ValueError
            If the transition is not legal according to the state graph.
        """
        old_state = self._state
        allowed = _LEGAL_TRANSITIONS.get(old_state, set())

        if new_state not in allowed:
            msg = (
                f"Illegal transition {old_state.name} -> {new_state.name} "
                f"for '{self._analysis_type}' pipeline. "
                f"Allowed targets: {sorted(s.name for s in allowed)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        # Apply the transition
        self._state = new_state
        self._context.update(context)

        logger.info(
            "[%s] %s -> %s  (elapsed=%.3fs)",
            self._analysis_type,
            old_state.name,
            new_state.name,
            self.elapsed,
        )

        # Notify subscribers
        self._notify_state_change(old_state, new_state)

    def report_progress(
        self, step_name: str, step_index: int, total_steps: int
    ) -> None:
        """Emit a progress event to all registered progress callbacks.

        Parameters
        ----------
        step_name : str
            Human-readable description of the current step.
        step_index : int
            Zero-based index of the current step.
        total_steps : int
            Total number of steps expected in this phase.
        """
        logger.debug(
            "[%s] Progress: '%s' (%d/%d)",
            self._analysis_type,
            step_name,
            step_index + 1,
            total_steps,
        )

        for cb in self._progress_callbacks:
            try:
                cb(step_name, step_index, total_steps)
            except Exception:
                logger.exception(
                    "Progress callback %r raised an exception", cb
                )

    def fail(self, error: BaseException) -> None:
        """Transition to :attr:`PipelineState.ERROR` with *error* stored in context.

        This is a convenience shortcut equivalent to::

            pipeline.transition(PipelineState.ERROR, error=error)

        Parameters
        ----------
        error : BaseException
            The exception that caused the failure.
        """
        logger.error(
            "[%s] Pipeline failed in state %s: %s",
            self._analysis_type,
            self._state.name,
            error,
        )
        self.transition(PipelineState.ERROR, error=error)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _notify_state_change(
        self, old_state: PipelineState, new_state: PipelineState
    ) -> None:
        """Call every registered state-change callback, catching errors."""
        for cb in self._state_callbacks:
            try:
                cb(old_state, new_state, self._context)
            except Exception:
                logger.exception(
                    "State-change callback %r raised an exception", cb
                )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AnalysisPipeline type={self._analysis_type!r} "
            f"state={self._state.name} elapsed={self.elapsed:.3f}s>"
        )


# ----------------------------------------------------------------------
# Convenience factory
# ----------------------------------------------------------------------

def create_pipeline(analysis_type: str) -> AnalysisPipeline:
    """Create and return a new :class:`AnalysisPipeline`.

    Parameters
    ----------
    analysis_type : str
        Label for the analysis (e.g. ``"series"``, ``"cross"``).

    Returns
    -------
    AnalysisPipeline
        A fresh pipeline in the ``IDLE`` state, ready to accept
        transitions.
    """
    return AnalysisPipeline(analysis_type)
