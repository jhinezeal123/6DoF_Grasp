from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

EPISODE_ID_RE = re.compile(r"^episode_(\d{6})$")
WAYPOINT_FILE_RE = re.compile(r"^episode_(\d{6})_wpts\.json$")


def format_episode_id(index: int) -> str:
    return f"episode_{int(index):06d}"


def format_waypoint_filename(index: int) -> str:
    return f"{format_episode_id(index)}_wpts.json"


def parse_episode_index(value: str | Path | int | None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if text == "":
        return None
    if text.isdigit():
        return int(text)
    name = Path(text).name
    m = EPISODE_ID_RE.match(name)
    if m:
        return int(m.group(1))
    m = WAYPOINT_FILE_RE.match(name)
    if m:
        return int(m.group(1))
    raise ValueError(
        "Episode selector must be an integer like '12', an id like 'episode_000012', "
        "or a waypoint filename like 'episode_000012_wpts.json'."
    )


def parse_waypoint_episode_index(path: str | Path) -> Optional[int]:
    name = Path(path).name
    match = WAYPOINT_FILE_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def normalized_waypoint_destination_name(episode_id: str | Path) -> str:
    name = Path(episode_id).name
    match = EPISODE_ID_RE.match(name)
    if not match:
        raise ValueError(f"Invalid episode id for waypoint naming: {episode_id!r}")
    return format_waypoint_filename(int(match.group(1)))
