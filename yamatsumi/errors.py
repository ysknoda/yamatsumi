"""Custom exception type shared across the yamatsumi package."""


class NEBError(RuntimeError):
    """Raised for unrecoverable errors in the CI-NEB workflow."""
