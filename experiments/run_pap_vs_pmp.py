#!/usr/bin/env python3
"""Standalone driver for the PAP-vs-PMP sweep. Built to be left running under tmux.

Does what sections 2-4 of `pap_vs_pmp_performance_computational_cost.ipynb` do, without a
kernel to lose: runs the grid for every model, scores the circuits, and writes the tables.
The analysis itself is imported from `pap_vs_pmp_tables`, which the notebook also imports, so
the two cannot drift.

**Resuming is the default.** Every cell of the grid is cached to its own JSON file, and a cell
counts as done only if its file exists *and* parses. Interrupt this at any point -- Ctrl-C, a
closed tmux window, an OOM, a reboot -- and re-run the same command: it recomputes only what is
missing. Writes are atomic, so a kill mid-write cannot leave a file that resume would trust.

    # see what is already done, without touching the GPU
    python experiments/run_pap_vs_pmp.py status

    # run (or resume) the whole sweep, logging to a file
    tmux new -s ipe
    python experiments/run_pap_vs_pmp.py run

    # just one model, or just the scoring phase
    python experiments/run_pap_vs_pmp.py run --models gpt2-small
    python experiments/run_pap_vs_pmp.py run --phase eval

    # rebuild the tables from cache (no GPU needed)
    python experiments/run_pap_vs_pmp.py tables

Detach from tmux with `Ctrl-b d`, reattach with `tmux attach -t ipe`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (os.path.join(REPO_ROOT, "src"), REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # `ipe` is used from src/, not pip-installed

from pap_vs_pmp_grid import MODELS, build_grid  # noqa: E402
from pap_vs_pmp_tables import all_tables, save_tables  # noqa: E402

WORKER = os.path.join(HERE, "pap_vs_pmp_worker.py")
DEFAULT_OUT = os.path.join(HERE, "pap_vs_pmp_performance_computational_cost")

# ----------------------------------------------------------------------------- logging


class Tee:
    """Write to the terminal and to a log file at once, so tmux scrollback is not the record."""

    def __init__(self, path: str | None):
        self.f = open(path, "a", buffering=1) if path else None
        if self.f:
            self.f.write(f"\n{'#' * 90}\n# session started {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n"
                         f"{'#' * 90}\n")

    def write(self, line: str = "") -> None:
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")

    def raw(self, chunk: str) -> None:
        """A chunk already carrying its own newline (subprocess output)."""
        sys.stdout.write(chunk)
        sys.stdout.flush()
        if self.f:
            self.f.write(chunk)

    def close(self) -> None:
        if self.f:
            self.f.close()


def hms(seconds: float) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60:d}m{s % 60:02d}s"


# ----------------------------------------------------------------------------- state


def read_json(path: str):
    """The record at `path`, or None if absent or unreadable (a kill mid-write)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return None


def model_state(model: str, out_dir: str, grid: list[dict]) -> dict:
    """What is on disk for one model: which cells are searched, which are scored."""
    slug = model.replace("/", "_")
    run_dir = os.path.join(out_dir, "runs", slug)
    score_dir = os.path.join(out_dir, "scores", slug)

    searched, failed, timed_out, corrupt, seconds = [], [], [], [], 0.0
    for cell in grid:
        rec = read_json(os.path.join(run_dir, cell["run_id"] + ".json"))
        if rec is None:
            if os.path.exists(os.path.join(run_dir, cell["run_id"] + ".json")):
                corrupt.append(cell["run_id"])
            continue
        searched.append(cell["run_id"])
        seconds += rec.get("seconds") or 0.0
        if rec.get("status") != "ok":
            failed.append(cell["run_id"])
        elif rec.get("timed_out"):
            timed_out.append(cell["run_id"])

    scored, eval_failed = [], []
    if os.path.isdir(score_dir):
        for cell in grid:
            rec = read_json(os.path.join(score_dir, cell["run_id"] + ".json"))
            if rec is None:
                continue
            scored.append(cell["run_id"])
            if rec.get("eval_status") != "ok":
                eval_failed.append(cell["run_id"])

    todo_search = [c["run_id"] for c in grid if c["run_id"] not in searched]
    # Only searches that produced a circuit can be scored.
    scorable = [r for r in searched if r not in failed]
    return {
        "model": model, "n_cells": len(grid),
        "searched": searched, "todo_search": todo_search, "failed": failed,
        "timed_out": timed_out, "corrupt": corrupt, "search_seconds": seconds,
        "scored": scored, "todo_eval": [r for r in scorable if r not in scored],
        "eval_failed": eval_failed,
    }


def print_status(out_dir: str, grid: list[dict], models: list[tuple], log: Tee,
                 max_time: float) -> list[dict]:
    states = [model_state(m, out_dir, grid) for m, _ in models]
    log.write("")
    log.write(f"{'model':30s} {'searched':>12s} {'scored':>12s} {'spent':>8s} "
              f"{'remaining (worst case)':>24s}")
    log.write("-" * 92)
    total_todo = 0
    for st in states:
        todo = len(st["todo_search"])
        total_todo += todo
        notes = []
        if st["timed_out"]:
            notes.append(f"{len(st['timed_out'])} timed out")
        if st["failed"]:
            notes.append(f"{len(st['failed'])} failed")
        if st["corrupt"]:
            notes.append(f"{len(st['corrupt'])} corrupt -> will redo")
        log.write(f"{st['model']:30s} {len(st['searched']):>5d}/{st['n_cells']:<6d} "
                  f"{len(st['scored']):>5d}/{len(st['searched']):<6d} "
                  f"{hms(st['search_seconds']):>8s} "
                  f"{hms(todo * max_time):>24s}"
                  + (("   " + ", ".join(notes)) if notes else ""))
    log.write("-" * 92)
    log.write(f"{'total':30s} "
              f"{sum(len(s['searched']) for s in states):>5d}/{len(grid)*len(models):<6d} "
              f"{sum(len(s['scored']) for s in states):>5d}"
              f"{'':<7s} {hms(sum(s['search_seconds'] for s in states)):>8s} "
              f"{hms(total_todo * max_time):>24s}")
    log.write("")
    return states


# ----------------------------------------------------------------------------- running


def run_worker(model: str, flags: list[str], out_dir: str, max_time: float,
               phase: str, retry_errors: bool, log: Tee, alloc_conf: str = "",
               extra: list[str] = ()) -> int:
    """One worker subprocess, streaming its output into the log as it goes."""
    cmd = [sys.executable, "-u", WORKER, "--model", model, "--out-dir", out_dir,
           "--max-time", str(max_time), "--phase", phase, *flags, *extra]
    if retry_errors:
        cmd.append("--retry-errors")
    env = {**os.environ,
           "PYTHONPATH": os.path.join(REPO_ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "PYTHONUNBUFFERED": "1",
           # Dump the Python stack on SIGSEGV/SIGABRT. A native crash otherwise leaves no trace
           # of *where* it happened, which is exactly what made the Qwen segfault undiagnosable.
           "PYTHONFAULTHANDLER": "1"}
    if alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
    log.write("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, cwd=HERE)
    try:
        for line in proc.stdout:
            log.raw(line)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    return proc.returncode


def cmd_run(a, grid: list[dict], models: list[tuple], log: Tee) -> int:
    log.write("=" * 92)
    log.write(f"PAP vs PMP sweep   |   {len(grid)} cells x {len(models)} model(s)   |   "
              f"budget {a.max_time:g}s per search")
    log.write(f"out: {a.out_dir}")
    log.write("=" * 92)
    print_status(a.out_dir, grid, models, log, a.max_time)

    interrupted = False
    for model, flags in models:
        st = model_state(model, a.out_dir, grid)
        if a.phase == "search" and not st["todo_search"] and not a.retry_errors:
            # Under "both" there is still the eval phase to run, so only skip a search-only pass.
            log.write(f"\n[{model}] all {len(grid)} searches cached, nothing to do")
            continue
        log.write("")
        log.write("=" * 92)
        log.write(f"  {model}   ({len(st['searched'])}/{len(grid)} searched, "
                  f"{len(st['scored'])} scored)")
        log.write("=" * 92)
        try:
            rc = run_worker(model, flags, a.out_dir, a.max_time, a.phase, a.retry_errors, log,
                            alloc_conf=a.alloc_conf)
        except KeyboardInterrupt:
            log.write("\n\n** interrupted **")
            interrupted = True
            break
        if rc != 0:
            if rc < 0:
                # Killed by a signal: no Python exception ran, so the cell in flight recorded
                # nothing itself. The worker picks that up from its in-flight marker next time.
                name = signal.Signals(-rc).name if -rc in set(s.value for s in signal.Signals) \
                    else "unknown"
                log.write(f"!! {model} was killed by signal {-rc} ({name}) -- not a Python error. "
                          f"The cell it was running is recorded as crashed on the next run.")
            else:
                log.write(f"!! {model} exited {rc}")
            log.write("   continuing with the next model (cached cells are kept; re-run to resume)")

    log.write("")
    log.write("=" * 92)
    log.write("final state")
    log.write("=" * 92)
    print_status(a.out_dir, grid, models, log, a.max_time)
    cmd_tables(a, log)
    if interrupted:
        log.write("\nresume with the same command:")
        log.write(f"  python {os.path.relpath(__file__, REPO_ROOT)} run"
                  + (f" --models {' '.join(m for m, _ in models)}"
                     if len(models) != len(MODELS) else ""))
        return 130
    return 0


def cmd_tables(a, log: Tee) -> int:
    tables = all_tables(a.out_dir)
    table = tables["table_full"]
    if table is None or table.empty:
        log.write("\nno scored circuits yet -- run the eval phase before expecting tables")
        return 0
    written = save_tables(tables, a.out_dir)
    log.write(f"\n{len(table)} scored rows")
    log.write("")
    with __import__("pandas").option_context("display.width", 220, "display.max_columns", 40,
                                             "display.max_rows", 300):
        log.write(tables["summary_modality"].round(3).to_string(index=False))
    log.write("")
    for p in written:
        rel = os.path.relpath(p, REPO_ROOT)
        log.write(f"  wrote {p if rel.startswith('..') else rel}")
    return 0


# ----------------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "status", "tables"], nargs="?", default="status")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--max-time", type=float, default=3600.0,
                   help="wall-clock budget per search, seconds (default: 1h)")
    p.add_argument("--models", nargs="+", default=None,
                   help="subset of model names to run (default: all three)")
    p.add_argument("--phase", default="both", choices=["search", "eval", "both"])
    p.add_argument("--retry-errors", action="store_true",
                   help="redo cells whose cached record is an error (e.g. a transient OOM)")
    p.add_argument("--alloc-conf", default="expandable_segments:True",
                   help="PYTORCH_CUDA_ALLOC_CONF for the workers. The default reduces the "
                        "fragmentation that made the Llama positional cells OOM with ~1.8GB "
                        "reserved-but-unallocated; pass '' to leave the allocator untouched.")
    p.add_argument("--log", default=None,
                   help="append output here (default: <out-dir>/sweep.log; '-' to disable)")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    log_path = None if a.log == "-" else (a.log or os.path.join(a.out_dir, "sweep.log"))
    log = Tee(log_path)

    models = MODELS
    if a.models:
        known = {m for m, _ in MODELS}
        unknown = [m for m in a.models if m not in known]
        if unknown:
            log.write(f"unknown model(s): {unknown}\nknown: {sorted(known)}")
            return 2
        models = [(m, f) for m, f in MODELS if m in a.models]

    grid = build_grid(a.max_time)
    try:
        if a.command == "status":
            print_status(a.out_dir, grid, models, log, a.max_time)
            for st in (model_state(m, a.out_dir, grid) for m, _ in models):
                if st["todo_search"]:
                    log.write(f"{st['model']}: {len(st['todo_search'])} search(es) remaining")
                    # A model that has not started has all 42 pending; listing them all buries
                    # the models that are genuinely part-done.
                    listed = st["todo_search"] if len(st["todo_search"]) <= 12 \
                        else st["todo_search"][:8]
                    for r in listed:
                        log.write(f"    {r}")
                    if len(listed) < len(st["todo_search"]):
                        log.write(f"    ... and {len(st['todo_search']) - len(listed)} more")
                if st["failed"]:
                    log.write(f"{st['model']}: {len(st['failed'])} failed "
                              f"(re-run with --retry-errors)")
                    for r in st["failed"]:
                        log.write(f"    {r}")
                if st["corrupt"]:
                    log.write(f"{st['model']}: {len(st['corrupt'])} unreadable cache file(s), "
                              f"will be recomputed")
                    for r in st["corrupt"]:
                        log.write(f"    {r}")
            return 0
        if a.command == "tables":
            return cmd_tables(a, log)
        return cmd_run(a, grid, models, log)
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
