"""
Scans an anc_log_*.csv file (produced by actual_anc_queue_run.py) for
episodes of negative ERLE, and classifies each one by:

  - duration:  BURST (short, < BURST_MAX_SECONDS) vs SUSTAINED (longer)
  - source:    did the reference mic also rise during the episode?
               (a broadband external source like traffic should raise
               both ref_rms and err_rms together; a localized transient
               like a door slam typically spikes err_rms without a
               matching, sustained rise in ref_rms)

Usage:
    python analyze_anc_log.py [anc_log_xxxxx.csv]

If no path is given, the most recent anc_log_*.csv in this directory is used.
"""

import csv
import sys
from pathlib import Path


NEGATIVE_RUN_MIN_ROWS = 1
BURST_MAX_SECONDS = 1.0
REF_RISE_FACTOR = 1.5
BACKGROUND_WINDOW_SECONDS = 10.0


def load_log(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                erle = float(row["erle_db"])
            except (KeyError, ValueError):
                continue
            rows.append({
                "t": float(row["elapsed_seconds"]),
                "erle": erle,
                "ref_rms": float(row["ref_rms"]),
                "err_rms": float(row["err_rms"]),
            })
    return rows


def find_negative_runs(rows):
    runs = []
    current = []
    for row in rows:
        if row["erle"] < 0:
            current.append(row)
        else:
            if len(current) >= NEGATIVE_RUN_MIN_ROWS:
                runs.append(current)
            current = []
    if len(current) >= NEGATIVE_RUN_MIN_ROWS:
        runs.append(current)
    return runs


def background_ref_rms(rows, run):
    start_t = run[0]["t"]
    nearby = [
        r["ref_rms"] for r in rows
        if start_t - BACKGROUND_WINDOW_SECONDS <= r["t"] < start_t and r["erle"] >= 0
    ]
    if not nearby:
        nearby = [r["ref_rms"] for r in rows if r["erle"] >= 0]
    return sum(nearby) / len(nearby) if nearby else 0.0


def classify_run(rows, run):
    start_t = run[0]["t"]
    end_t = run[-1]["t"]
    duration = end_t - start_t

    mean_ref = sum(r["ref_rms"] for r in run) / len(run)
    mean_err = sum(r["err_rms"] for r in run) / len(run)
    mean_erle = sum(r["erle"] for r in run) / len(run)

    bg_ref = background_ref_rms(rows, run)

    shape = "BURST" if duration <= BURST_MAX_SECONDS else "SUSTAINED"

    if bg_ref > 0 and mean_ref >= bg_ref * REF_RISE_FACTOR:
        source = "ref mic also rose  -> likely external/broadband (traffic-like)"
    else:
        source = "ref mic stayed flat -> likely localized/transient (door-like)"

    return {
        "start": start_t,
        "end": end_t,
        "duration": duration,
        "mean_erle": mean_erle,
        "mean_ref_rms": mean_ref,
        "mean_err_rms": mean_err,
        "background_ref_rms": bg_ref,
        "shape": shape,
        "source": source,
    }


def main():
    if len(sys.argv) >= 2:
        path = Path(sys.argv[1])
    else:
        candidates = sorted(Path(__file__).parent.glob("anc_log_*.csv"))
        if not candidates:
            print("No anc_log_*.csv files found. Usage: python analyze_anc_log.py <log.csv>")
            return
        path = candidates[-1]
        print(f"No path given -- using most recent log: {path.name}\n")

    rows = load_log(path)
    if not rows:
        print("No usable rows found (log empty, or run never left baseline).")
        return

    runs = find_negative_runs(rows)
    if not runs:
        print("No negative-ERLE episodes found in this log -- ERLE stayed >= 0 throughout.")
        return

    total_time = rows[-1]["t"] - rows[0]["t"]
    negative_time = sum(run[-1]["t"] - run[0]["t"] for run in runs)

    print(f"Log: {path.name}")
    print(f"Duration analysed: {total_time:.1f}s | Time spent ERLE<0: {negative_time:.1f}s "
          f"({100 * negative_time / total_time:.1f}%)\n")
    print(f"Found {len(runs)} negative-ERLE episode(s):\n")

    for run in runs:
        info = classify_run(rows, run)
        print(
            f"[{info['start']:7.1f}s - {info['end']:7.1f}s]  "
            f"dur={info['duration']:5.1f}s  "
            f"meanERLE={info['mean_erle']:6.2f}dB  "
            f"{info['shape']:9s}  "
            f"ref={info['mean_ref_rms']:.5f} (bg={info['background_ref_rms']:.5f})  "
            f"-> {info['source']}"
        )


if __name__ == "__main__":
    main()
