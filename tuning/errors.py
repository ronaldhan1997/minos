"""Errors for the practice-mode tuner."""


class TuningError(Exception):
    """A configuration, manifest, or operator error. The study should stop or skip."""


class ScoringVersionAbort(TuningError):
    """The practice output reported a different scoring version than the study pin."""
