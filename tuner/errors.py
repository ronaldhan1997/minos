"""Exceptions raised by the tuner. CLI commands turn these into exit codes."""


class TunerError(Exception):
    """A problem the operator can fix from the message."""


class StudyConflict(TunerError):
    """The named study already exists under a different scoring contract."""


class ExportError(TunerError):
    """A config must not be written."""


class PrepareError(TunerError):
    """Practice data could not be resolved."""


class InputError(TunerError):
    """BAM, reference, truth, or region failed the pre-run checks."""
