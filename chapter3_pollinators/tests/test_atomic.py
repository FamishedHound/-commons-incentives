"""Atomic file replacement while other processes read the file."""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from chapter3_pollinators.experiment import atomic


def test_replace_retries_while_the_target_is_busy(tmp_path, monkeypatch) -> None:
    target = tmp_path / "progress.json"
    target.write_text("old")
    tmp = tmp_path / "progress.json.tmp"
    tmp.write_text("new")
    calls = {"n": 0}
    real_replace = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "in use")
        real_replace(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky)
    atomic.replace_file(tmp, target)
    assert calls["n"] == 3 and target.read_text() == "new" and not tmp.exists()


def test_replace_gives_up_after_the_retry_period(tmp_path, monkeypatch) -> None:
    def always_busy(src, dst):
        raise PermissionError(13, "in use")

    monkeypatch.setattr(atomic.os, "replace", always_busy)
    started = time.monotonic()
    with pytest.raises(PermissionError):
        atomic.replace_file(tmp_path / "a.tmp", tmp_path / "a", retry_seconds=0.2)
    assert 0.2 <= time.monotonic() - started < 2.0


@pytest.mark.skipif(sys.platform != "win32", reason="open files block replacement on Windows only")
def test_replace_waits_for_a_reader_to_close_the_file(tmp_path) -> None:
    target = tmp_path / "status.json"
    target.write_text("old")
    tmp = tmp_path / "status.json.tmp"
    tmp.write_text("new")
    reader = open(target, encoding="utf-8")  # noqa: SIM115 - held open on purpose
    with pytest.raises(PermissionError):
        os.replace(tmp, target)
    closer = threading.Timer(0.3, reader.close)
    closer.start()
    atomic.replace_file(tmp, target)
    closer.join()
    assert target.read_text() == "new"
