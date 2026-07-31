"""Global, cross-process GPU leases.

WHY THIS EXISTS
---------------
`tune_ccas.py` and `sweep_ccas.py` both support `--shard i/n` so several launcher processes can
split one work list. Each launcher used to enforce `--max-concurrent` *locally*, over its own copy
of `--gpus`. Two shards started with `--gpus 0,1 --max-concurrent 2` therefore ran FOUR training
jobs, two per physical GPU. GPU memory was never the binding constraint (7.9 GB of 80 GB per job),
but 4 jobs x 8 dataloader workers exhausted the host cgroup and the kernel OOM-killed
`pt_data_worker`:

    oom-kill:constraint=CONSTRAINT_MEMCG ... task=pt_data_worker
    Memory cgroup out of memory: Killed process (pt_data_worker) anon-rss:29603...

Five runs were lost that way, silently: the `.log` simply stops mid-epoch with no traceback, and no
result JSON is written, so the cells look "pending" rather than "failed".

A per-launcher limit cannot fix this, because the launchers do not know about each other. The
constraint that actually matters is physical — *one job per GPU* — so it is enforced where the
processes do share state: the filesystem.

HOW
---
One lock file per GPU, published by `link()` from a fully-written temp file. `link()` fails with
EEXIST atomically, and — crucially — the lock file already contains its owner's pid the instant it
becomes visible.

The obvious O_CREAT|O_EXCL version is subtly wrong and was caught by a real multi-process test:
creating the file and writing the pid into it are two syscalls, so a racing caller can open the
freshly created but still EMPTY file, read no pid, conclude the lease is stale, and take it. Three
processes racing for two GPUs handed GPU 1 to two of them. Publishing an already-complete file
removes the window entirely.

A launcher that dies without releasing leaves a stale file; the next caller reads the pid, finds no
such process, and reclaims it. So a crash costs one extra poll interval, never a permanently wedged
GPU. A lock file that cannot be parsed is treated as HELD, never as free: refusing to run is
cheap, double-booking a GPU is not.

WHOSE pid GOES IN THE FILE
-------------------------
The lease records the pid of the *training job*, not the launcher. That is the physically correct
invariant: a GPU is busy exactly while a `run_ccas.py` process is on it. A launcher can be killed,
backgrounded or restarted without freeing a GPU that is still computing, and a launcher that dies
between jobs does not wedge anything. Callers therefore `acquire()` with their own pid (closing the
race before `Popen`), then `adopt()` the child's pid the moment it exists.

Concurrency is then bounded by `len(gpus)` no matter how many launchers, shards or humans are
running. `--max-concurrent` remains as a per-launcher courtesy limit, but it is no longer what
protects the host.
"""
import errno
import os
import uuid
from pathlib import Path

LOCK_DIR = Path(os.environ.get("CCAS_GPU_LOCK_DIR", "/tmp/ccas_gpu_locks"))


def _alive(pid):
    """Is `pid` a live process? Signal 0 checks existence without delivering anything."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                       # exists, owned by someone else
        return True
    return True


def _path(gpu):
    return LOCK_DIR / f"gpu{gpu}.lock"


def _read_holder(p):
    """-> pid, or -1 if the file is missing/empty/garbage."""
    try:
        return int(p.read_text().strip() or -1)
    except (ValueError, OSError):
        return -1


def _publish(p, pid):
    """Atomically create `p` containing `pid`. -> True if we won the race.

    link() is the atomic step, and it publishes a file that is already complete, so no reader can
    ever observe a lock file without an owner in it.
    """
    tmp = p.parent / f"{p.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(str(pid))
    try:
        os.link(tmp, p)
        return True
    except FileExistsError:
        return False
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def acquire(gpu, pid=None):
    """Try to take the lease on `gpu`. -> True if this process now holds it.

    Never blocks: callers poll, so a busy GPU is an ordinary negative answer, not an error.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    pid = os.getpid() if pid is None else pid
    p = _path(gpu)
    if _publish(p, pid):
        return True
    holder = _read_holder(p)
    if holder == pid:                              # re-entrant: we already hold it
        return True
    if holder < 0 or _alive(holder):
        # Either a live owner, or a lock we cannot read. Both mean "not ours to take".
        return False
    # Stale lease from a launcher that died. Reclaim it -- but publish again under link(), so if
    # two reclaimers race, exactly one wins.
    try:
        os.unlink(p)
    except FileNotFoundError:
        pass
    return _publish(p, pid)


def release(gpu, pid=None):
    """Drop the lease if we hold it. Releasing someone else's lease is a no-op, by design."""
    pid = os.getpid() if pid is None else pid
    p = _path(gpu)
    if _read_holder(p) != pid:
        return False
    try:
        os.unlink(p)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
    return True


def adopt(gpu, pid, prev_pid=None):
    """Hand an already-held lease to `pid` (the training subprocess we just spawned).

    Called immediately after Popen. Until this lands the lease names the launcher, which is alive,
    so the GPU cannot be double-booked in the gap. Afterwards the lease tracks the job itself, so
    the GPU stays reserved even if the launcher goes away, and is reclaimed as soon as the job
    exits -- with or without a result JSON.
    """
    prev_pid = os.getpid() if prev_pid is None else prev_pid
    p = _path(gpu)
    holder = _read_holder(p)
    if holder != prev_pid:
        return False
    p.write_text(str(pid))
    return True


def acquire_any(gpus, pid=None):
    """First free GPU from `gpus`, or None. Order is preserved so slot 0 is preferred."""
    for g in gpus:
        if acquire(g, pid=pid):
            return g
    return None


def holders(gpus):
    """-> {gpu: pid} for currently held leases. For logging and diagnostics only."""
    out = {}
    for g in gpus:
        h = _read_holder(_path(g))
        if h > 0:
            out[g] = h
    return out
