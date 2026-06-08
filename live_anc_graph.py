"""
Live demo dashboard for the DANC exhibition stand.

Two panels, side by side:

  LEFT  -- a scrolling trend graph: the live noise level at the error mic
           (red line, system running) against the level that same mic
           measured before cancellation started (dashed grey "baseline"
           line). Green shading shows exactly when/how much the live level
           dips below baseline -- proof the system keeps working over time.

  RIGHT -- a big, glanceable level meter: two bars, "BEFORE" (system off,
           the baseline) vs "NOW" (system on, smoothed live reading), plus
           a bold percentage readout that flips green/red for quieter/
           louder. This is the "wow, look how much shorter that bar is"
           panel -- no axes or trends to interpret, just a height comparison.

Run actual_anc_queue_run.py first (it streams readings to anc_log_*.csv),
then run this script alongside it in a second terminal/window:

    DISPLAY=:0 python live_anc_graph.py

It always follows the newest anc_log_*.csv and picks up new rows as they are
written, so nothing needs to be restarted or re-pointed.
"""

import csv
import os
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib

# The backend is selectable so the same script works two ways:
#   * On the Pi with a monitor (default):
#         DISPLAY=:0 python live_anc_graph.py
#   * Remotely from another machine, in a browser:
#         ANC_GRAPH_BACKEND=WebAgg python live_anc_graph.py
#     WebAgg serves the live figure as a small web page. Over VSCode
#     Remote-SSH the port is auto-forwarded, so it opens in your laptop's
#     browser -- no monitor, no X11/XQuartz needed.
_BACKEND = os.environ.get("ANC_GRAPH_BACKEND", "TkAgg")
matplotlib.use(_BACKEND)
if _BACKEND.lower() == "webagg":
    matplotlib.rcParams["webagg.open_in_browser"] = False  # headless Pi
    matplotlib.rcParams["webagg.port"] = int(os.environ.get("ANC_GRAPH_PORT", "8988"))
    matplotlib.rcParams["webagg.port_retries"] = 50
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button


WINDOW_SECONDS = 15.0     # how much history the trend panel shows at once
POLL_INTERVAL_MS = 200    # how often the dashboard refreshes
METER_SMOOTHING_ROWS = 18  # ~1.5s of rows -- keeps the meter bar from jittering

# Shared file used to ask the ANC process to switch on/off. The graph's toggle
# button writes the requested state here; actual_anc_queue_run.py reads it each
# loop. Must match ANC_CONTROL_FILE in that script.
ANC_CONTROL_FILE = Path(__file__).parent / "anc_control.txt"


def request_anc_state(enabled):
    """Ask the ANC process to turn cancellation on/off, by writing the shared
    control file atomically (temp file + rename). Never raises -- if the write
    fails the button click is simply a no-op.
    """
    try:
        tmp = ANC_CONTROL_FILE.with_suffix(".txt.tmp")
        tmp.write_text("on" if enabled else "off")
        os.replace(tmp, ANC_CONTROL_FILE)
    except Exception:
        pass


def find_log_path():
    if len(sys.argv) >= 2:
        return Path(sys.argv[1])

    print("Looking for an anc_log_*.csv to follow "
          "(start actual_anc_queue_run.py if it isn't running yet)...")
    while True:
        candidates = sorted(
            Path(__file__).parent.glob("anc_log_*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return candidates[-1]
        time.sleep(0.5)


class LogTail:
    """Follows a growing CSV file, yielding only complete rows as dicts.

    Reads line-by-line and rewinds on a partial line (one written but not yet
    newline-terminated by the still-running ANC process), so nothing is ever
    skipped or mis-parsed mid-write.
    """

    def __init__(self, path):
        self._file = open(path, newline="")
        header_line = self._readline_complete()
        while header_line is None:
            time.sleep(0.1)
            header_line = self._readline_complete()
        self._fields = next(csv.reader([header_line]))

    def _readline_complete(self):
        pos = self._file.tell()
        line = self._file.readline()
        if not line.endswith("\n"):
            self._file.seek(pos)
            return None
        return line

    def poll(self):
        rows = []
        while True:
            line = self._readline_complete()
            if line is None:
                return rows
            values = next(csv.reader([line]))
            if len(values) == len(self._fields):
                rows.append(dict(zip(self._fields, values)))


def main():
    log_path = find_log_path()
    print(f"\nFollowing: {log_path.name}")
    print("Waiting for the run to finish measuring its baseline "
          "(speaker stays silent for a few seconds at the start)...\n")

    tail = LogTail(log_path)

    times = []
    err_levels = []
    recent_err = deque(maxlen=METER_SMOOTHING_ROWS)
    baseline_rms = None
    fill = [None]
    anc_status = [None]   # last ANC status string seen in the log

    fig = plt.figure(figsize=(15, 6.5))
    try:
        fig.canvas.manager.set_window_title("DANC -- live cancellation demo")
    except Exception:
        pass
    fig.suptitle("Directional Active Noise Cancellation -- live performance",
                 fontsize=16, fontweight="bold", y=0.975)

    # Big, glanceable ANC ON/OFF banner -- driven by the status the ANC process
    # actually logs (so it reflects the real state, including SAFETY MUTE), not
    # just what the button requested.
    anc_banner = fig.text(
        0.5, 0.915, "ANC: starting...", ha="center", va="center",
        fontsize=17, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="tab:gray", edgecolor="none"),
    )

    gs = fig.add_gridspec(1, 2, width_ratios=[5, 2], wspace=0.32,
                          left=0.06, right=0.97, top=0.83, bottom=0.16)
    ax_trend = fig.add_subplot(gs[0])
    ax_meter = fig.add_subplot(gs[1])

    # ---- toggle button: works in the native window AND the remote browser ----
    ax_button = fig.add_axes([0.74, 0.03, 0.22, 0.07])
    toggle_button = Button(ax_button, "Turn ANC ON", color="0.85", hovercolor="0.7")
    current_on = [None]   # last known actual state, from the logged status

    def on_toggle(_event):
        # Flip relative to the last known state (default to turning ON if we
        # don't know it yet). request_anc_state writes the shared control file;
        # the ANC process applies it on its next loop.
        request_anc_state(not (current_on[0] or False))

    toggle_button.on_clicked(on_toggle)

    # ---- left panel: scrolling trend ----
    err_line, = ax_trend.plot(
        [], [], color="tab:red", linewidth=2.5,
        label="Noise reaching the listener NOW (system running)",
    )
    baseline_line = ax_trend.axhline(
        0, color="tab:gray", linewidth=2.5, linestyle="--",
        label="Noise level with the system OFF (baseline)",
    )
    baseline_line.set_visible(False)

    ax_trend.set_xlabel("Time (seconds since cancellation started)")
    ax_trend.set_ylabel("Sound level at the listener's ear")
    ax_trend.set_title("Over time", fontsize=13)
    ax_trend.grid(True, alpha=0.3)
    ax_trend.legend(loc="upper right", fontsize=9)

    # ---- right panel: big before/now level meter ----
    bars = ax_meter.bar(
        ["BEFORE\n(system off)", "NOW\n(system on)"],
        [0, 0], width=0.6,
        color=["tab:gray", "tab:green"], edgecolor="black", linewidth=1.3,
    )
    pct_text = ax_meter.text(
        0.5, 1.10, "", transform=ax_meter.transAxes,
        ha="center", va="bottom", fontsize=24, fontweight="bold",
    )
    ax_meter.set_title("Right now", fontsize=13)
    ax_meter.set_ylabel("Sound level at the listener's ear")
    ax_meter.set_yticks([])
    for spine in ("top", "right"):
        ax_meter.spines[spine].set_visible(False)

    def update(_frame):
        nonlocal baseline_rms

        for row in tail.poll():
            try:
                t = float(row["elapsed_seconds"])
                err = float(row["err_rms"])
            except (KeyError, ValueError):
                continue

            status = row.get("status", "")
            if status:
                anc_status[0] = status

            if baseline_rms is None:
                raw = row.get("baseline_rms", "")
                if raw:
                    try:
                        baseline_rms = float(raw)
                    except ValueError:
                        baseline_rms = None
                    if baseline_rms is not None:
                        baseline_line.set_ydata([baseline_rms, baseline_rms])
                        baseline_line.set_visible(True)
                        bars[0].set_height(baseline_rms)
                        print(f"\nBaseline captured: {baseline_rms:.5f} "
                              "(this is what the error mic hears with the system OFF)\n")

            times.append(t)
            err_levels.append(err)
            recent_err.append(err)

        # ---- ANC ON/OFF banner + toggle button label (from logged status) ----
        status = anc_status[0]
        if status == "ANC OFF":
            current_on[0] = False
            anc_banner.set_text("ANC OFF  --  no cancellation")
            anc_banner.get_bbox_patch().set_facecolor("tab:red")
            toggle_button.label.set_text("Turn ANC ON")
        elif status == "SAFETY MUTE":
            current_on[0] = True
            anc_banner.set_text("ANC ON  --  safety mute")
            anc_banner.get_bbox_patch().set_facecolor("tab:orange")
            toggle_button.label.set_text("Turn ANC OFF")
        elif status in ("ANC running", "ANC started"):
            current_on[0] = True
            anc_banner.set_text("ANC ON  --  cancelling")
            anc_banner.get_bbox_patch().set_facecolor("tab:green")
            toggle_button.label.set_text("Turn ANC OFF")
        elif status:
            anc_banner.set_text(f"{status}...")
            anc_banner.get_bbox_patch().set_facecolor("tab:gray")

        if not times:
            return err_line, baseline_line, bars[0], bars[1], pct_text

        smoothed_now = sum(recent_err) / len(recent_err)

        if baseline_rms is not None:
            quieter_pct = 100.0 * (1.0 - smoothed_now / baseline_rms)
            # Kept short and padded to a fixed width on purpose -- see the
            # matching note in actual_anc_queue_run.py: a line longer than
            # the terminal wraps onto a second row, and "\r" then only
            # rewinds to the start of that wrapped row, breaking the
            # single-line-refresh effect and printing a wall of text.
            line = (
                f"t={times[-1]:6.1f}s | "
                f"base={baseline_rms:.4f} | "
                f"now={smoothed_now:.4f} | "
                f"{quieter_pct:+5.1f}% quieter"
            )
            print(f"\r{line:<60}", end="", flush=True)

            bars[1].set_height(smoothed_now)
            if smoothed_now < baseline_rms:
                bars[1].set_color("tab:green")
                pct_text.set_text(f"{quieter_pct:.0f}% QUIETER")
                pct_text.set_color("tab:green")
            else:
                bars[1].set_color("tab:red")
                pct_text.set_text(f"{-quieter_pct:.0f}% LOUDER")
                pct_text.set_color("tab:red")

            ax_meter.set_ylim(0, max(baseline_rms, smoothed_now) * 1.35)

        # ---- update trend panel ----
        cutoff = times[-1] - WINDOW_SECONDS
        while len(times) > 1 and times[0] < cutoff:
            times.pop(0)
            err_levels.pop(0)

        err_line.set_data(times, err_levels)

        if fill[0] is not None:
            fill[0].remove()
            fill[0] = None
        if baseline_rms is not None and len(times) > 1:
            fill[0] = ax_trend.fill_between(
                times, err_levels, baseline_rms,
                where=[e < baseline_rms for e in err_levels],
                color="tab:green", alpha=0.25, interpolate=True,
            )

        ax_trend.set_xlim(times[0], times[-1] + 0.5)
        peak = max(err_levels + ([baseline_rms] if baseline_rms else [1e-6]))
        ax_trend.set_ylim(0, peak * 1.25)

        return err_line, baseline_line, bars[0], bars[1], pct_text

    ani = animation.FuncAnimation(
        fig, update, interval=POLL_INTERVAL_MS, cache_frame_data=False,
    )

    if _BACKEND.lower() == "webagg":
        port = matplotlib.rcParams["webagg.port"]
        print(f"\nWebAgg mode: open the live graph in your browser at "
              f"http://localhost:{port}/")
        print("(In VSCode Remote-SSH the port is forwarded automatically -- "
              "look for the pop-up,\n or the PORTS tab, then open it on your "
              "Mac. Ctrl+C here to stop.)\n")

    plt.show()


if __name__ == "__main__":
    main()
