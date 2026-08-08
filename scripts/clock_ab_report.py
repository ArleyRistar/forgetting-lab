#!/usr/bin/env python3
"""Compare the clock-cap A/B arms (LAB-NOTES open item 2).

Hypothesis under test: a LOWER SM clock cap raises *average* throughput by
preventing the boost -> overheat -> hard-throttle oscillation seen at bring-up.
So the headline number is wall-clock for a fixed step count, not peak s/it.

Stdlib only, and run with system python3 on purpose: `uv run` reinstalls the
local flab package into the shared .venv, which must not happen while a run is
using it.

Usage: python3 scripts/clock_ab_report.py
"""
import re
import statistics as st
from pathlib import Path

ARMS = [("a1200", "1200 MHz"), ("a1000", "1000 MHz")]
STEP_RE = re.compile(r"(\d+)/(\d+) \[(\d+:\d+(?::\d+)?)<")


def _secs(clock: str) -> int:
    parts = [int(x) for x in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def steps(tag: str) -> list[tuple[int, int]]:
    """(step, elapsed_seconds), deduped and ordered.

    The log holds two tqdm bars — `Loading weights: 106/290` and the training
    bar — and a naive regex silently merges them, which yields a 290-step arm
    with a negative s/it. Bars are separated by their total, and the training
    bar is the one with by far the most distinct entries.

    Step 1 is dropped: tqdm's first estimate is always nonsense (quirk 3).
    """
    path = Path(f"/tmp/ab-{tag}.log")
    if not path.is_file():
        return []
    bars: dict[int, dict[int, int]] = {}
    for m in STEP_RE.finditer(path.read_text(errors="replace").replace("\r", "\n")):
        step, total, clock = int(m.group(1)), int(m.group(2)), m.group(3)
        bars.setdefault(total, {}).setdefault(step, _secs(clock))
    if not bars:
        return []
    seen = max(bars.values(), key=len)
    return [(k, seen[k]) for k in sorted(seen) if k > 1]


def telemetry(tag: str) -> dict:
    path = Path(f"/tmp/ab-{tag}.csv")
    if not path.is_file():
        return {}
    temp, power, clock, util = [], [], [], []
    for i, line in enumerate(path.read_text().splitlines()):
        if i == 0:  # nvidia-smi's first power.draw read returns ~751 W (quirk 3)
            continue
        try:
            t, p, c, u = (x.strip() for x in line.split(","))
            temp.append(float(t)); power.append(float(p))
            clock.append(float(c)); util.append(float(u))
        except ValueError:
            continue
    if not temp:
        return {}
    return {
        "n": len(temp),
        "temp_mean": st.mean(temp), "temp_max": max(temp),
        "power_mean": st.mean(power),
        "clock_mean": st.mean(clock),
        "clock_sd": st.pstdev(clock) if len(clock) > 1 else 0.0,
        "clock_min": min(clock), "clock_max": max(clock),
        "util_mean": st.mean(util),
    }


def window(pairs: list[tuple[int, int]], lo: int, hi: int) -> float | None:
    """Mean s/it strictly between step lo and hi, from elapsed-time deltas."""
    sel = [(s, e) for s, e in pairs if lo <= s <= hi]
    if len(sel) < 2:
        return None
    return (sel[-1][1] - sel[0][1]) / (sel[-1][0] - sel[0][0])


def main() -> None:
    rows = {}
    for tag, label in ARMS:
        p = steps(tag)
        if not p:
            print(f"!! arm {tag}: no step data at /tmp/ab-{tag}.log")
            continue
        n_last, n_first = p[-1][0], p[0][0]
        rows[tag] = {
            "label": label,
            "steps": n_last,
            "wall_s": p[-1][1],
            "overall": (p[-1][1] - p[0][1]) / (n_last - n_first),
            "cold": window(p, 2, 50),
            "soaked": window(p, 100, 150),
            **telemetry(tag),
        }

    if len(rows) < 2:
        print("\nIncomplete: need both arms before this comparison means anything.")
        return

    def fmt(v, spec=".2f"):
        return "n/a" if v is None else format(v, spec)

    print(f"\n{'metric':<28}" + "".join(f"{r['label']:>14}" for r in rows.values()))
    print("-" * (28 + 14 * len(rows)))
    for key, name, spec in [
        ("steps", "steps completed", "d"),
        ("wall_s", "wall-clock (s)", "d"),
        ("overall", "mean s/it (overall)", ".2f"),
        ("cold", "mean s/it (steps 2-50)", ".2f"),
        ("soaked", "mean s/it (steps 100-150)", ".2f"),
        ("temp_mean", "temp mean (C)", ".1f"),
        ("temp_max", "temp max (C)", ".0f"),
        ("power_mean", "power mean (W)", ".1f"),
        ("clock_mean", "SM clock mean (MHz)", ".0f"),
        ("clock_sd", "SM clock sd (MHz)", ".1f"),
        ("clock_min", "SM clock min (MHz)", ".0f"),
        ("util_mean", "GPU util mean (%)", ".1f"),
    ]:
        print(f"{name:<28}" + "".join(f"{fmt(r.get(key), spec):>14}" for r in rows.values()))

    a, b = rows.get("a1200"), rows.get("a1000")
    if not (a and b and a["overall"] and b["overall"]):
        return
    if a["steps"] != b["steps"]:
        print(f"\nARMS INCOMPLETE — {a['steps']} vs {b['steps']} steps. The comparison "
              "below is provisional;\nre-run once both arms have finished.")
    delta = (a["overall"] - b["overall"]) / a["overall"] * 100
    verdict = "FASTER" if delta > 0 else "SLOWER"
    print(f"\n1000 MHz is {abs(delta):.1f}% {verdict} than 1200 MHz on mean s/it.")
    if all(x.get("cold") and x.get("soaked") for x in (a, b)):
        print(f"Within-arm derate  1200: {a['soaked']/a['cold']:.2f}x   "
              f"1000: {b['soaked']/b['cold']:.2f}x  (soaked / cold)")
        print("\nCONFOUND: arm 2 started at 50 C, arm 1 at 39 C — the 900 s cooldown\n"
              "timed out before reaching its 45 C target. The bias is directional:\n"
              "it handicaps 1000 MHz, so a 1000 MHz win is conservative and a\n"
              "1000 MHz loss is inconclusive and needs a rerun from equal temps.")


if __name__ == "__main__":
    main()
