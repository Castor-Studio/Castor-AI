class ModuleNotFoundError(LookupError):
    """Raised when requested AI module is not installed."""


class ModuleLoadError(RuntimeError):
    """Raised when an installed AI module cannot be constructed."""
