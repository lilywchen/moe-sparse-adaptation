"""Guards on the GPU lease. Torch-free and GPU-free, so they run anywhere.

The failure these prevent is expensive and silent: over-subscribed GPUs exhaust the host cgroup,
the kernel kills dataloader workers, and the affected runs vanish without a traceback and without a
result JSON — so they read as "pending", not "failed". Five real runs were lost that way.

Liveness is injected rather than taken from the real process table, so the tests describe the
lease's logic instead of depending on which pids happen to exist on the machine running them.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "gpulease", ROOT / "moe_shift" / "utils" / "gpulease.py")
gl = importlib.util.module_from_spec(_spec)
sys.modules["gpulease"] = gl
_spec.loader.exec_module(gl)

LIVE = set()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Fresh lock dir, and a process table we control."""
    LIVE.clear()
    monkeypatch.setattr(gl, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(gl, "_alive", lambda pid: pid in LIVE)


def live(pid):
    LIVE.add(pid)
    return pid


# ------------------------------------------------------------------ exclusion
def test_second_acquire_of_the_same_gpu_fails():
    assert gl.acquire("0", pid=live(1111)) is True
    assert gl.acquire("0", pid=live(2222)) is False


def test_acquire_is_reentrant_for_the_holder():
    assert gl.acquire("0", pid=live(1111)) is True
    assert gl.acquire("0", pid=1111) is True


def test_acquire_any_prefers_the_first_free_gpu():
    assert gl.acquire_any(["0", "1"], pid=live(1111)) == "0"
    assert gl.acquire_any(["0", "1"], pid=live(2222)) == "1"
    assert gl.acquire_any(["0", "1"], pid=live(3333)) is None


def test_two_shards_cannot_exceed_the_gpu_count():
    """THE regression: 2 launchers x max_concurrent 2 must yield 2 jobs, not 4."""
    gpus = ["0", "1"]
    granted = []
    for shard in (100, 200):                        # two launcher processes
        for job in range(2):                        # each willing to run 2 concurrently
            g = gl.acquire_any(gpus, pid=live(shard * 10 + job))
            if g is not None:
                granted.append(g)
    assert sorted(granted) == ["0", "1"], granted


# ------------------------------------------------------------------ release
def test_release_frees_the_gpu():
    gl.acquire("0", pid=live(1111))
    assert gl.release("0", pid=1111) is True
    assert gl.acquire("0", pid=live(2222)) is True


def test_a_process_cannot_release_someone_elses_lease():
    gl.acquire("0", pid=live(1111))
    assert gl.release("0", pid=2222) is False
    assert gl.acquire("0", pid=live(2222)) is False


# ------------------------------------------------------------------ crash recovery
def test_stale_lease_from_a_dead_launcher_is_reclaimed():
    """A crashed launcher must not wedge a GPU forever."""
    gl.acquire("0", pid=live(1111))
    LIVE.discard(1111)                              # the launcher dies
    assert gl.acquire("0", pid=live(2222)) is True


def test_live_holder_is_never_evicted():
    gl.acquire("0", pid=live(1111))
    assert gl.acquire("0", pid=live(2222)) is False


# ------------------------------------------------------------------ adoption
def test_adopt_transfers_the_lease_to_the_training_job():
    gl.acquire("0", pid=live(1111))                 # launcher takes it
    assert gl.adopt("0", pid=live(4242), prev_pid=1111) is True
    assert gl.holders(["0"]) == {"0": 4242}


def test_lease_survives_the_launcher_dying_mid_run():
    """The point of adoption: a job still on the GPU keeps the GPU."""
    gl.acquire("0", pid=live(1111))
    gl.adopt("0", pid=live(4242), prev_pid=1111)
    LIVE.discard(1111)                              # launcher dies, training continues
    assert gl.acquire("0", pid=live(2222)) is False


def test_lease_is_reclaimed_once_the_training_job_exits():
    gl.acquire("0", pid=live(1111))
    gl.adopt("0", pid=live(4242), prev_pid=1111)
    LIVE.discard(4242)                              # job finished (or was OOM-killed)
    assert gl.acquire("0", pid=live(2222)) is True


def test_adopt_refuses_when_we_are_not_the_holder():
    gl.acquire("0", pid=live(1111))
    assert gl.adopt("0", pid=4242, prev_pid=9999) is False
    assert gl.holders(["0"]) == {"0": 1111}


def test_holders_reports_the_owning_pids():
    gl.acquire("0", pid=live(1111))
    assert gl.holders(["0", "1"]) == {"0": 1111}


# ---------------------------------------------------- the race a sequential test cannot see
def test_a_lock_file_without_a_pid_is_treated_as_held_not_stale():
    """The exact O_EXCL bug: file created, pid not written yet. Must NOT look free."""
    gl.LOCK_DIR.mkdir(parents=True, exist_ok=True)
    gl._path("0").write_text("")                    # a lease caught mid-publication
    assert gl.acquire("0", pid=live(2222)) is False


def test_an_unparseable_lock_file_is_treated_as_held():
    gl.LOCK_DIR.mkdir(parents=True, exist_ok=True)
    gl._path("0").write_text("not-a-pid")
    assert gl.acquire("0", pid=live(2222)) is False


def test_published_lock_file_always_contains_its_owner(tmp_path):
    """Whatever a racing reader observes, it must never be an ownerless file."""
    gl.acquire("0", pid=live(1111))
    assert gl._path("0").read_text().strip() == "1111"


def test_real_processes_racing_for_two_gpus_get_exactly_two(tmp_path):
    """The regression that unit tests with a fake process table cannot reproduce.

    Eight real OS processes contend for two GPUs at the same moment, and each winner stays alive
    while the others are still trying — so liveness comes from the real process table, not a stub.
    Before `link()`-based publication this handed the same GPU to more than one of them, because a
    racer could observe a lock file that existed but did not yet contain a pid.
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    start = ctx.Barrier(8)
    lock_dir = str(tmp_path / "racelocks")
    src = str(ROOT / "moe_shift" / "utils" / "gpulease.py")

    def worker(q, start, d, s):
        import importlib.util as iu
        import pathlib
        import time
        sp = iu.spec_from_file_location("gl2", s)
        m = iu.module_from_spec(sp)
        sp.loader.exec_module(m)
        m.LOCK_DIR = pathlib.Path(d)
        start.wait()                                # everyone pounces together
        q.put(m.acquire_any(["0", "1"]))
        time.sleep(2)                               # winners stay alive during the race

    procs = [ctx.Process(target=worker, args=(q, start, lock_dir, src)) for _ in range(8)]
    for p in procs:
        p.start()
    got = [q.get(timeout=30) for _ in range(8)]
    for p in procs:
        p.join(30)
    won = [g for g in got if g is not None]
    assert sorted(won) == ["0", "1"], got
