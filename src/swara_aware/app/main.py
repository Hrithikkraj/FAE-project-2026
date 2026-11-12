"""Start the Swara-Aware live app.

Usage (from the swara-aware folder):
    Windows:      $env:PYTHONPATH="src"; python -m swara_aware
    macOS/Linux:  PYTHONPATH=src python -m swara_aware
    or simply:    python scripts/run_app.py

Options:
    --wav FILE        feed a recording through the pipeline in real time instead of the mic
    --loop            with --wav, repeat the file
    --sa NOTE         initial Sa as a note name (default D3)
    --method NAME     fft_peak, fft_hps, autocorr or yin (default yin)
    --screenshot PNG  (developer) save a window screenshot after --quit-after seconds
    --quit-after S    (developer) close automatically after S seconds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import Config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m swara_aware",
                                     description="Swara-Aware: live swara pitch feedback.")
    parser.add_argument("--wav", help="analyse this audio file in real time instead of the mic")
    parser.add_argument("--loop", action="store_true", help="repeat the --wav file")
    parser.add_argument("--sa", default="D3", help="initial Sa as a note name, e.g. C#3")
    parser.add_argument("--method", default="yin",
                        choices=["fft_peak", "fft_hps", "autocorr", "yin"])
    parser.add_argument("--screenshot", help=argparse.SUPPRESS)
    parser.add_argument("--quit-after", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, build the window, run the Qt event loop."""
    args = parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")  # session plots are saved to files, never shown in a window

    try:
        import pyqtgraph as pg
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        print(f"The live app needs PySide6 and pyqtgraph ({exc}).\n"
              "Install them with:  pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    cfg = Config()  # files are resampled to cfg.sample_rate when they are opened
    if args.wav and not Path(args.wav).is_file():
        print(f"File not found: {args.wav}", file=sys.stderr)
        sys.exit(1)

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    pg.setConfigOptions(antialias=True)
    window = MainWindow(cfg, wav_path=args.wav, loop=args.loop, sa_note=args.sa,
                        method=args.method)
    window.show()
    if args.wav:
        QTimer.singleShot(300, window.start_listening)  # file demo starts by itself
    if args.quit_after is not None:
        def finish() -> None:
            if args.screenshot:
                window.grab().save(args.screenshot)
            window.close()
        QTimer.singleShot(int(args.quit_after * 1000), finish)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
