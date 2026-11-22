from importlib.metadata import entry_points

from mlkit.use_cases.base import UseCase

ENTRY_POINT_GROUP = "mlkit.use_cases"
USE_CASES: dict[str, UseCase] = {}
_DISCOVERED = False


def register_use_case(use_case: UseCase) -> None:
    """Register one use-case contract."""
    if not isinstance(use_case, UseCase):
        raise TypeError(
            "Registered use cases must be mlkit.use_cases.UseCase instances."
        )
    if use_case.name in USE_CASES:
        raise ValueError(f"Use case '{use_case.name}' is already registered.")
    USE_CASES[use_case.name] = use_case


def discover_use_cases() -> None:
    """Load use cases exposed by installed packages."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        use_case = entry_point.load()
        register_use_case(use_case)
    _DISCOVERED = True


def get_use_case(name: str) -> UseCase:
    """Return a registered use case by its explicit name."""
    discover_use_cases()
    try:
        return USE_CASES[name]
    except KeyError as exc:
        available = ", ".join(sorted(USE_CASES))
        raise ValueError(
            f"Unknown use case '{name}'. Available use cases: {available}."
        ) from exc
