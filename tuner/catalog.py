"""Practice-catalog coverage for a study.

``configs/sample_id.md`` is the list of practice ids. A chromosome filter
only sees ids that are already staged, so a study can silently score a few
local BAMs while the catalog still has dozens of ids. ``tune`` refuses that
unless ``--allow-partial-catalog`` is set.
"""

from __future__ import annotations

from tuner.errors import TunerError


def catalog_status(listed: list[str], ready_ids: list[str]) -> tuple[str, list[str]]:
    """Return a one-line summary and the catalog ids that are not runnable."""
    ordered = list(dict.fromkeys(item.lower() for item in listed))
    ready = {item.lower() for item in ready_ids}
    missing = [item for item in ordered if item not in ready]
    present = len(ordered) - len(missing)
    if not ordered:
        return "No sample-id catalog. Using staged practice samples.", []
    return (
        f"catalog {len(ordered)} ids, runnable {present}, missing {len(missing)}",
        missing,
    )


def assert_selected_ids(selected: list[str], ready_ids: list[str]) -> str:
    """An explicit UUID list is the whole study. The other catalog ids stay out."""
    ordered = list(dict.fromkeys(item.lower() for item in selected))
    ready = {item.lower() for item in ready_ids}
    missing = [item for item in ordered if item not in ready]
    if missing:
        raise TunerError(
            "Selected sample id is not runnable: "
            + ", ".join(missing)
            + ". Stage it with prepare before tune."
        )
    noun = "sample id" if len(ordered) == 1 else "sample ids"
    return f"selected {len(ordered)} {noun} for this study"


def assert_catalog_coverage(
    listed: list[str],
    ready_ids: list[str],
    *,
    allow_partial: bool,
) -> str:
    summary, missing = catalog_status(listed, ready_ids)
    if not missing or allow_partial:
        return summary
    preview = ", ".join(missing[:5])
    if len(missing) > 5:
        preview += f" (+{len(missing) - 5} more)"
    raise TunerError(
        f"{summary}. tune will not search a subset of the sample catalog. "
        f"Missing: {preview}. "
        "Run prepare with PLATFORM_URL set and no --samples to download the catalog, "
        "or pass --allow-partial-catalog to search only the runnable ids."
    )
