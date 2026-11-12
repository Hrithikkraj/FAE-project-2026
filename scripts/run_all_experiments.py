"""Run every Stage 5 experiment and write results/*.csv, results/*.png and results/RESULTS.md.

Audio concepts: reproducible evaluation (fixed seed), accuracy in cents, SNR, latency.
The tanpura drone is never played here (it would leak into a microphone).

Usage (from the swara-aware folder):
    python scripts/run_all_experiments.py              # everything (about a minute or two)
    python scripts/run_all_experiments.py --only E1,E5 # a subset (RESULTS.md lists only those)
    python scripts/run_all_experiments.py --quick      # smaller sizes, for a fast check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from swara_aware.config import Config  # noqa: E402
from swara_aware.eval import accuracy, latency, noise, ornament_eval, sensitivity, voice  # noqa: E402
from swara_aware.eval.common import RESULTS_DIR  # noqa: E402
from swara_aware.eval.report import write_report  # noqa: E402

FULL = {
    "E1": lambda cfg, out: accuracy.run(cfg, out),
    "E2": lambda cfg, out: noise.run(cfg, out),
    "E3": lambda cfg, out: latency.run(cfg, out),
    "E4": lambda cfg, out: voice.run(cfg, out),
    "E5": lambda cfg, out: ornament_eval.run(cfg, out),
    "E6": lambda cfg, out: sensitivity.run(cfg, out),
}
QUICK = {
    "E1": lambda cfg, out: accuracy.run(cfg, out, n_freqs=8, trials=2),
    "E2": lambda cfg, out: noise.run(cfg, out, n_freqs=4, trials=2),
    "E3": lambda cfg, out: latency.run(cfg, out, n_frames=60),
    "E4": lambda cfg, out: voice.run(cfg, out),
    "E5": lambda cfg, out: ornament_eval.run(cfg, out, n_phrases=3),
    "E6": lambda cfg, out: sensitivity.run(cfg, out, n_freqs=4, trials=2),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Swara-Aware evaluation.")
    parser.add_argument("--only", help="comma-separated experiment keys, e.g. E1,E3")
    parser.add_argument("--quick", action="store_true", help="small sizes for a fast check")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="output folder")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles can't print every symbol
        sys.stdout.reconfigure(errors="replace")

    table = QUICK if args.quick else FULL
    keys = [k.strip().upper() for k in args.only.split(",")] if args.only else list(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    results, start = [], time.perf_counter()
    for key in keys:
        t0 = time.perf_counter()
        print(f"{key} ...", end=" ", flush=True)
        result = table[key](cfg, out)
        results.append(result)
        status = "done" if result.ran else "skipped"
        print(f"{status} in {time.perf_counter() - t0:.1f} s. {result.finding}")
    write_report(results, cfg, out / "RESULTS.md", time.perf_counter() - start)
    print(f"\nWrote {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
