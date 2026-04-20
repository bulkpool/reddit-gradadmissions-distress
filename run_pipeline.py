#!/usr/bin/env python3
"""
run_pipeline.py — Execute the full CS598 pipeline for both subreddits.

Injects the SUBREDDIT variable into each notebook's config cell and runs it
via nbconvert. Source notebooks are never modified. Notebook output is
streamed to per-step log files in pipeline_logs/.

Usage:
    ~/venvs/jupyter/bin/python3 run_pipeline.py
    ~/venvs/jupyter/bin/python3 run_pipeline.py --subreddits mscs
    ~/venvs/jupyter/bin/python3 run_pipeline.py --start-from 04
    ~/venvs/jupyter/bin/python3 run_pipeline.py --skip-comparison
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.resolve()
NB_DIR   = ROOT / 'notebooks'
LOG_DIR  = ROOT / 'pipeline_logs'
JUPYTER  = Path.home() / 'venvs' / 'jupyter' / 'bin' / 'jupyter'

# ── Pipeline order ─────────────────────────────────────────────────────────────
PIPELINE_NBS = [
    '01_clean_corpus.ipynb',
    '03_exposure_labels_v2.ipynb',
    '04_panel_scores.ipynb',
    '05_collect_community_breadth.ipynb',
    '06_did_analysis_v2.ipynb',
    '08_alt_analysis.ipynb',
]
COMPARISON_NB = '07_comparison_analysis.ipynb'

# Short display names for the progress bar
NB_SHORT = {
    '01_clean_corpus.ipynb':              '01 clean corpus',
    '03_exposure_labels_v2.ipynb':        '03 exposure labels',
    '04_panel_scores.ipynb':              '04 panel scores',
    '05_collect_community_breadth.ipynb': '05 breadth (API)',
    '06_did_analysis_v2.ipynb':           '06 DiD analysis',
    '08_alt_analysis.ipynb':              '08 alt analysis',
    '07_comparison_analysis.ipynb':       '07 comparison',
}

WIDTH = 68   # terminal width for the UI

# ── Terminal helpers ───────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f'{h}h {m:02d}m {s:02d}s'
    if m:
        return f'{m}m {s:02d}s'
    return f'{s}s'


def _bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total else 0
    return '█' * filled + '░' * (width - filled)


def print_header(total_steps: int, subreddits: list[str]) -> None:
    print()
    print('┌' + '─' * (WIDTH - 2) + '┐')
    title = f'CS598 Pipeline  ·  {" + ".join(f"r/{s}" for s in subreddits)}'
    print(f'│  {title:<{WIDTH - 4}}│')
    print(f'│  {total_steps} steps total                                              │')
    print('└' + '─' * (WIDTH - 2) + '┘')


def print_step_start(step: int, total: int, label: str) -> None:
    bar  = _bar(step - 1, total)
    pct  = int(100 * (step - 1) / total)
    line = f'  Step {step}/{total}  [{bar}]  {pct:3d}%'
    print()
    print('─' * WIDTH)
    print(line)
    print(f'  ▶  {label}')
    print('─' * WIDTH)


def print_step_done(step: int, total: int, label: str,
                    elapsed: float, ok: bool) -> None:
    bar    = _bar(step, total)
    pct    = int(100 * step / total)
    status = '✓' if ok else '✗'
    dur    = _fmt_duration(elapsed)
    print(f'  {status}  {label}  ({dur})')
    print(f'  [{bar}]  {pct:3d}%  —  {step}/{total} done')


def print_final(total: int, elapsed: float, failed: list[str]) -> None:
    print()
    print('═' * WIDTH)
    if not failed:
        bar = _bar(total, total)
        print(f'  ✓  Pipeline complete  [{bar}]  100%')
        print(f'     Total time: {_fmt_duration(elapsed)}')
    else:
        print(f'  ✗  Pipeline finished with {len(failed)} failure(s):')
        for f in failed:
            print(f'       • {f}')
        print(f'     Total time: {_fmt_duration(elapsed)}')
        print(f'     Logs in:    {LOG_DIR}/')
    print('═' * WIDTH)
    print()


# ── Spinner ────────────────────────────────────────────────────────────────────

class Spinner:
    """Live spinner with elapsed time, running in a background thread."""

    _FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, message: str) -> None:
        self._msg     = message
        self._running = False
        self._thread: threading.Thread | None = None
        self._start   = 0.0

    def start(self) -> None:
        self._start   = time.time()
        self._running = True
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join()
        elapsed = time.time() - self._start
        # Clear the spinner line
        sys.stdout.write('\r' + ' ' * (WIDTH + 4) + '\r')
        sys.stdout.flush()
        return elapsed

    def _spin(self) -> None:
        i = 0
        while self._running:
            elapsed = time.time() - self._start
            dur     = _fmt_duration(elapsed)
            frame   = self._FRAMES[i % len(self._FRAMES)]
            line    = f'\r  {frame}  {self._msg}  {dur} elapsed ...'
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(0.12)
            i += 1


# ── Core logic ─────────────────────────────────────────────────────────────────

def inject_subreddit(nb_path: Path, subreddit: str, out_path: Path) -> None:
    """Write a copy of nb_path with the SUBREDDIT variable set to subreddit."""
    nb      = json.loads(nb_path.read_text())
    patched = False

    for cell in nb['cells']:
        if patched or cell['cell_type'] != 'code':
            continue
        src = cell['source']
        if isinstance(src, str):
            lines, was_str = src.splitlines(keepends=True), True
        else:
            lines, was_str = list(src), False

        joined = ''.join(lines)
        if not re.search(r"SUBREDDIT\s*=\s*'", joined) or '# change to' not in joined:
            continue

        new_lines = []
        for line in lines:
            if re.match(r"\s*SUBREDDIT\s*=\s*'", line) and '# change to' in line:
                indent = len(line) - len(line.lstrip())
                new_lines.append(
                    ' ' * indent
                    + f"SUBREDDIT = '{subreddit}'"
                    + '   # injected by run_pipeline.py\n'
                )
            else:
                new_lines.append(line)

        cell['source'] = ''.join(new_lines) if was_str else new_lines
        patched = True

    if not patched:
        print(f'  ⚠  No SUBREDDIT config cell in {nb_path.name} — running as-is')

    out_path.write_text(json.dumps(nb, indent=1))


def run_notebook(nb_path: Path, out_path: Path, log_path: Path) -> tuple[bool, float]:
    """Execute notebook; stream output to log_path. Returns (ok, elapsed_sec).

    nb_path must be inside NB_DIR so that Path('..').resolve() inside the
    notebook kernel correctly resolves to the project root.
    """
    spinner = Spinner(nb_path.name)
    spinner.start()

    with open(log_path, 'w') as log_fh:
        result = subprocess.run(
            [
                str(JUPYTER), 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--output', str(out_path),
                '--ExecutePreprocessor.timeout=7200',
                '--ExecutePreprocessor.kernel_name=python3',
                str(nb_path),
            ],
            stdout=log_fh,
            stderr=log_fh,
        )

    elapsed = spinner.stop()
    ok      = result.returncode == 0

    if not ok:
        print(f'  ✗  Error — see log: {log_path.name}')

    return ok, elapsed


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run the CS598 pipeline for one or more subreddits.'
    )
    parser.add_argument(
        '--subreddits', nargs='+', default=['gradadmissions', 'mscs'],
    )
    parser.add_argument(
        '--start-from', default=None, metavar='NN',
        help='Skip notebooks whose two-digit prefix < NN (e.g. "04")',
    )
    parser.add_argument(
        '--skip-comparison', action='store_true',
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Temp notebooks must sit directly inside NB_DIR (same level as real
    # notebooks) so that Path('..').resolve() inside each kernel correctly
    # resolves to the project root — not in a subdirectory of NB_DIR.
    tmp_prefix    = f'_run_{run_ts}'
    tmp_notebooks: list[Path] = []   # cleaned up in finally

    # Build the ordered step list so we know total up front
    steps: list[tuple[str, str]] = []   # (subreddit, nb_name) — '' sr means comparison
    for sr in args.subreddits:
        for nb_name in PIPELINE_NBS:
            if args.start_from and nb_name[:2] < args.start_from:
                continue
            steps.append((sr, nb_name))
    if not args.skip_comparison:
        steps.append(('', COMPARISON_NB))

    total      = len(steps)
    failed: list[str] = []
    wall_start = time.time()

    print_header(total, args.subreddits)

    try:
        for step_idx, (sr, nb_name) in enumerate(steps, start=1):
            if sr:
                label = f'r/{sr}  ·  {NB_SHORT.get(nb_name, nb_name)}'
            else:
                label = f'comparison  ·  {NB_SHORT.get(nb_name, nb_name)}'

            print_step_start(step_idx, total, label)

            src_nb   = NB_DIR / nb_name
            slug     = f'{sr}_{nb_name}' if sr else nb_name
            # Both tmp and output notebooks live directly in NB_DIR
            tmp_nb   = NB_DIR / f'{tmp_prefix}_{slug}'
            out_nb   = NB_DIR / f'{tmp_prefix}_{slug.replace(".ipynb", "_out.ipynb")}'
            log_path = LOG_DIR / f'{run_ts}_{slug.replace(".ipynb", ".log")}'
            tmp_notebooks.extend([tmp_nb, out_nb])

            if sr:
                inject_subreddit(src_nb, sr, tmp_nb)
            else:
                shutil.copy2(src_nb, tmp_nb)

            ok, elapsed = run_notebook(tmp_nb, out_nb, log_path)

            print_step_done(step_idx, total, label, elapsed, ok)
            if not ok:
                failed.append(label)

    finally:
        for f in tmp_notebooks:
            f.unlink(missing_ok=True)

    print_final(total, time.time() - wall_start, failed)

    if failed:
        print(f'  Logs saved to: {LOG_DIR}/')
        sys.exit(1)


if __name__ == '__main__':
    main()
