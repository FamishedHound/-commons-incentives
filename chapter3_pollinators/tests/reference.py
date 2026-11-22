"""Load the reference implementation of the Pollinators' Game for equivalence tests.

The equivalence tests run the package side by side with the reference source
code of the game (the ``all_in_one`` module folder) and require bit-identical
behaviour on fixed seeds.  Point ``CHAPTER3_REFERENCE_SRC`` at that folder; the
tests that need it are skipped when it is not set.

The reference modules are imported from a temporary copy with bytecode
writing switched off, so the reference folder itself is only ever read.  The
copy runs headless, with ``gym`` served by ``gymnasium``.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import types
from pathlib import Path

ENV_VAR = "CHAPTER3_REFERENCE_SRC"

#: The reference modules the tests use, in import order.
MODULES = (
    "GlobalParamsGame",
    "LandCell",
    "Grid",
    "Agent",
    "MADDPG_agent",
    "PolinattorsProcessor",
    "AgentProcessor",
    "ActionProcessor",
    "EnvironmentalManager",
    "EconomyManager",
    "gym_pollinator_game",
    "memory",
    "model",
    "MADDPG",
    "goverment_agent",
)


def reference_source() -> Path | None:
    """The reference ``all_in_one`` folder, or ``None`` when it is not configured."""
    value = os.environ.get(ENV_VAR)
    if not value:
        return None
    path = Path(value)
    return path if (path / "gym_pollinator_game.py").is_file() else None


def _install_gym_alias() -> None:
    import gymnasium

    alias = types.ModuleType("gym")
    alias.Env = gymnasium.Env
    alias.spaces = gymnasium.spaces
    sys.modules["gym"] = alias


def _install_headless_pygame() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    pygame.image.load = lambda *args, **kwargs: pygame.Surface((1, 1))


class _ListStack:
    """``np.stack`` that accepts a generator, as NumPy < 1.24 did."""

    def __init__(self, numpy_module: types.ModuleType) -> None:
        self._np = numpy_module

    def __getattr__(self, name: str):
        return getattr(self._np, name)

    def stack(self, arrays, *args, **kwargs):
        return self._np.stack(list(arrays), *args, **kwargs)


def load_reference(workdir: Path) -> types.SimpleNamespace:
    """Copy the reference modules into ``workdir`` and import them."""
    source = reference_source()
    if source is None:
        raise FileNotFoundError(f"{ENV_VAR} is not set to the reference source folder")
    package_dir = workdir / "all_in_one"
    package_dir.mkdir(parents=True, exist_ok=True)
    for file in source.glob("*.py"):
        shutil.copy2(file, package_dir / file.name)

    sys.dont_write_bytecode = True
    _install_gym_alias()
    _install_headless_pygame()
    for entry in (str(package_dir), str(workdir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    modules = {name: importlib.import_module(name) for name in MODULES}
    regulator = modules["goverment_agent"]
    regulator.np = _ListStack(regulator.np)
    return types.SimpleNamespace(**modules)
