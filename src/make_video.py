"""Experiment video: the three update rules trained side by side, then the usable learning rates.

Scenes: a title card; Optdigits networks with two hidden layers trained by BP+SGD, ProxProp and
ProxProp-all at learning rates 0.01, 10 and 100, epoch by epoch, with their real training logs and
the predictions on 60 validation digits; the usable learning rates of all nine settings; a closing
card. Data: results/demo_trace.json (src/trace_demo.py) and docs/data/sweep.js (src/demo_data.py).
Renders 1920x1080 frames with matplotlib and encodes them with ffmpeg.
Usage: python src/make_video.py [--out docs/assets/experiments.mp4] [--workers 4] [--at 10,20]
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, DPI, FPS = 1920, 1080, 100, 30
BG, PANEL, INK, MUTED, EDGE = "#000000", "#0e0e0e", "#f2f2f2", "#9a9a9a", "#3a3a3a"
GOOD, BAD, WARN = "#3fb950", "#f85149", "#d29922"
METHODS = ["bp_sgd", "prox_sgd", "prox_all"]
NAME = {"bp_sgd": "BP+SGD", "prox_sgd": "ProxProp", "prox_all": "ProxProp-all"}
COL = {"bp_sgd": "#f08a5d", "prox_sgd": "#5aa0e0", "prox_all": "#bfe0ff"}
MARK = {"bp_sgd": "o", "prox_sgd": "s", "prox_all": "D"}
SERIF, MONO = "Times New Roman", "Consolas"
plt.rcParams.update({"font.family": "serif", "font.serif": [SERIF, "DejaVu Serif"], "mathtext.fontset": "stix",
                     "text.color": INK, "axes.edgecolor": EDGE, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "savefig.facecolor": BG})

# timeline (seconds): title, three training races, the sweep, the closing card
RACE = 12.5  # 1 s at epoch 0, 9 s for 100 epochs, 2.5 s holding the result
SCENES = [("title", 0.0, 4.0, None), ("race", 4.0, 4.0 + RACE, 0.01), ("race", 4.0 + RACE, 4.0 + 2 * RACE, 10.0),
          ("race", 4.0 + 2 * RACE, 4.0 + 3 * RACE, 100.0), ("sweep", 41.5, 59.5, None), ("end", 59.5, 65.5, None)]
DURATION = SCENES[-1][2]
CAPTION = {
    0.01: ("Same network, data and seed: only the update rule differs. Learning rate η = 0.01.",
           "At a small learning rate BP+SGD learns, while the damped proximal steps learn too little."),
    10.0: ("Learning rate η = 10.",
           "At η = 10 BP+SGD’s weights blow up to |W| ≈ 10²⁹ and it predicts one class; "
           "both proximal networks reach 98%."),
    100.0: ("Learning rate η = 100.",
            "At η = 100 ProxProp fails numerically in its first epoch; only ProxProp-all still trains (97.9%)."),
}

DATA = {}


def load():
    tr = json.load(open(os.path.join(ROOT, "results", "demo_trace.json")))
    runs = {(r["method"], r["lr"]): r for r in tr["runs"]}
    t = open(os.path.join(ROOT, "docs", "data", "sweep.js")).read()
    sweep = json.loads(t[t.index("{"):t.rindex("}") + 1])
    DATA.update(trace=tr, runs=runs, sweep=sweep)


# ------------------------------------------------------------------ helpers
def fade_overlay(fig, a):
    if a > 0:
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color="black", alpha=min(1, a), zorder=1000))


def scene_fade(t, t0, t1, d=0.45):
    return max(0.0, 1 - (t - t0) / d, 1 - (t1 - t) / d)


def ftext(fig, x, y, s, size, color=INK, family=SERIF, ha="center", va="center", weight="normal"):
    fig.text(x, y, s, fontsize=size, color=color, family=family, ha=ha, va=va, weight=weight)


def digit_grid(digits, preds, show):
    """60 validation digits, six rows of ten, framed green if predicted correctly and red if not."""
    cell, pad = 40, 4
    img = np.zeros((6 * cell, 10 * cell, 3))
    green, red = np.array([63, 185, 80]) / 255, np.array([248, 81, 73]) / 255
    for k, d in enumerate(digits):
        r, c = divmod(k, 10)
        y0, x0 = r * cell, c * cell
        if show:
            color = green if preds[k] == d["label"] else red
            img[y0:y0 + cell, x0:x0 + cell] = color * 0.85
            img[y0 + pad - 1:y0 + cell - pad + 1, x0 + pad - 1:x0 + cell - pad + 1] = color * 0.18
        g = np.kron(np.array(d["x"], dtype=float).reshape(8, 8) / 16.0, np.ones((4, 4)))
        img[y0 + pad:y0 + pad + 32, x0 + pad:x0 + pad + 32] = np.maximum(
            img[y0 + pad:y0 + pad + 32, x0 + pad:x0 + pad + 32], g[..., None] * np.array([1, 1, 1]))
    return img


def status_of(run, lr, epoch, done):
    if run["failure"] and epoch >= run["failure"]["epoch"] + 1:
        return f"NUMERICAL FAILURE in epoch {run['failure']['epoch'] + 1}", BAD
    if not done:
        return "training…", MUTED
    acc = 100 * run["final_val_acc"]
    if acc >= 90:
        return f"DONE · val. accuracy {acc:.1f}%", GOOD
    if lr <= 0.1:  # a small rate: the network was still learning when training stopped
        return f"LEARNED TOO LITTLE · {acc:.1f}%", WARN
    if acc < 50:
        return f"COLLAPSED · val. accuracy {acc:.1f}%", BAD
    return f"DONE · val. accuracy {acc:.1f}%", WARN


def short_log(run, lr, epoch):
    lines = []
    h = run
    n = min(epoch, len(h["val_acc"]))
    for e in range(max(0, n - 9), n):
        lines.append(f"epoch {e + 1:3d}  loss {h['train_loss'][e]:8.2e}  acc {100 * h['val_acc'][e]:5.1f}%  "
                     f"|W| {h['w_norm'][e]:7.1e}")
    if run["failure"] and epoch >= run["failure"]["epoch"] + 1:
        lines.append(f"epoch {run['failure']['epoch'] + 1:3d}  {run['failure']['reason']}: proximal solve failed")
        lines.append("-> scored as a constant predictor")
    return lines[-9:]


# ------------------------------------------------------------------ scenes
def draw_title(fig, t, t0, t1):
    ftext(fig, 0.5, 0.60, "Learning-Rate Sensitivity of Backpropagation", 58)
    ftext(fig, 0.5, 0.50, "Does Proximal Backpropagation widen the range of good learning rates, or only move it?", 28,
          color=MUTED)
    ftext(fig, 0.5, 0.38, "Zekai Yu, Tingshuo Fan, Jiacheng Chen  ·  University of California, Davis", 24)
    ftext(fig, 0.5, 0.33, "ECS 171 Machine Learning, Final Project", 22, color=MUTED)
    fade_overlay(fig, scene_fade(t, t0, t1, 0.6))


def draw_race(fig, t, t0, t1, lr):
    tr = DATA["trace"]
    p = (t - t0 - 1.0) / 9.0
    epoch = 0 if p <= 0 else min(100, int(np.floor(p * 100)) + 1)
    done = t - t0 >= 10.0
    lr_txt = f"{lr:g}"
    ftext(fig, 0.035, 0.945, f"η = {lr_txt}", 46, ha="left")
    ftext(fig, 0.035, 0.895, "Optdigits · 2 hidden layers · seed 0 · 100 epochs", 20, color=MUTED, ha="left")
    ftext(fig, 0.5, 0.945, f"epoch {epoch:3d} / 100", 30, color=MUTED, family=MONO)
    for k, m in enumerate(METHODS):
        run = DATA["runs"][(m, lr)]
        x0 = 0.03 + k * 0.325
        cx = x0 + 0.145
        ftext(fig, cx, 0.835, NAME[m], 36)
        fig.add_artist(Line2D([cx - 0.04, cx + 0.04], [0.806, 0.806], transform=fig.transFigure, color=COL[m], lw=4))
        s, c = status_of(run, lr, epoch, done)
        ftext(fig, cx, 0.775, s, 21, color=c, family=MONO)
        term = fig.add_axes([x0, 0.53, 0.29, 0.215])
        term.set_facecolor(PANEL)
        term.set_xticks([])
        term.set_yticks([])
        for sp in term.spines.values():
            sp.set_color(EDGE)
        log = short_log(run, lr, epoch)
        term.text(0.025, 0.95, "\n".join(log) if log else f"$ train --rule {m} --lr {lr_txt}", transform=term.transAxes,
                  family=MONO, fontsize=11.6, color="#d8d8d8", va="top", ha="left", linespacing=1.45)
        grid = fig.add_axes([x0 + 0.02, 0.215, 0.25, 0.29])
        grid.imshow(digit_grid(tr["digits"], run["preds"][epoch], epoch > 0), interpolation="nearest")
        grid.axis("off")
        acc = fig.add_axes([x0 + 0.03, 0.105, 0.23, 0.085])
        acc.set_facecolor(BG)
        for sp in acc.spines.values():
            sp.set_color(EDGE)
        acc.set_xlim(0, 100)
        acc.set_ylim(0, 1.02)
        acc.set_yticks([0, 0.5, 1])
        acc.set_yticklabels(["0", "50%", "100%"], fontsize=11)
        acc.set_xticks([0, 50, 100])
        acc.tick_params(labelsize=11, length=2)
        va = run["val_acc"]
        n = min(epoch, len(va))
        if n:
            acc.plot(np.arange(1, n + 1), va[:n], color=COL[m], lw=2.2)
        if run["failure"] and epoch >= run["failure"]["epoch"] + 1:
            e0 = run["failure"]["epoch"] + 1
            acc.plot([e0, max(e0, epoch)], [run["final_val_acc"]] * 2, color=BAD, lw=2, ls=(0, (3, 2)))
        acc.text(1.0, 1.08, "val. accuracy", transform=acc.transAxes, ha="right", va="bottom", fontsize=12, color=MUTED)
    caption = CAPTION[lr][1] if done else CAPTION[lr][0]
    ftext(fig, 0.5, 0.035, caption, 25)
    fade_overlay(fig, scene_fade(t, t0, t1))


def draw_sweep(fig, t, t0, t1):
    sw = DATA["sweep"]
    lrs = sw["lrs"]
    n = len(lrs)
    cursor = (n - 1) * min(1.0, max(0.0, (t - t0 - 1.0) / 11.0))
    done = t - t0 >= 12.0
    ftext(fig, 0.5, 0.94, "Learning rates at which each update rule trains well, in all nine settings", 34)
    ftext(fig, 0.5, 0.895, "final validation accuracy within 2 points of the best run, mean of 5 seeds", 21, color=MUTED)
    for k, m in enumerate(METHODS):  # legend
        x = 0.36 + k * 0.11
        fig.add_artist(Line2D([x, x + 0.025], [0.85, 0.85], transform=fig.transFigure, color=COL[m], lw=7))
        ftext(fig, x + 0.03, 0.85, NAME[m], 20, ha="left")
    archs = ["shallow", "medium", "deep"]
    for d, (ds, dsn) in enumerate([("drybean", "Dry Bean"), ("optdigits", "Optdigits"), ("har", "HAR")]):
        ax = fig.add_axes([0.075 + d * 0.31, 0.19, 0.28, 0.56])
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(EDGE)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(10.8, -0.8)
        ax.set_xticks([0, 4, 8, 12])
        ax.set_xticklabels(["$10^{-4}$", "$10^{-2}$", "$1$", "$10^{2}$"], fontsize=17)
        if d == 0:
            ax.set_yticks([1, 5, 9])
            ax.set_yticklabels(["1 hidden\nlayer", "2 hidden\nlayers", "3 hidden\nlayers"], fontsize=16, color=INK)
        else:
            ax.set_yticks([])
        ax.tick_params(length=0)
        for i in range(n):
            ax.axvline(i, color="#1c1c1c", lw=1, zorder=0)
        ax.set_title(dsn, fontsize=28, color=INK, pad=10)
        ax.set_xlabel("learning rate η", fontsize=19, color=INK)
        for g, arch in enumerate(archs):
            st = sw["settings"][f"{ds}/{arch}"]["methods"]
            for j, m in enumerate(METHODS):
                y = 4 * g + j
                r = st[m]
                ok = [u and i <= cursor + 1e-9 for i, u in enumerate(r["usable"])]
                i = 0
                while i < n:
                    if ok[i]:
                        k2 = i
                        while k2 + 1 < n and ok[k2 + 1]:
                            k2 += 1
                        if k2 > i:
                            ax.plot([i, k2], [y, y], color=COL[m], lw=7, solid_capstyle="round", zorder=3)
                        else:
                            ax.plot(i, y, "o", ms=7, color=COL[m], zorder=3)
                        i = k2 + 1
                    else:
                        i += 1
                for i, run in enumerate(r["runs"]):
                    if run["fails"] and i <= cursor + 1e-9:
                        ax.plot(i, y, "x", ms=8, mew=1.8, color="#8a8a8a", zorder=2)
        if not done:
            ax.axvline(cursor, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=4)
    if done:
        ftext(fig, 0.5, 0.075, "ProxProp moves the usable range up by 0.5–2 decades but widens it in only 3 of 9 settings.", 25)
        ftext(fig, 0.5, 0.035, "ProxProp-all widens it in 8 of 9 settings, with no numerical failure.", 25)
    else:
        ftext(fig, 0.5, 0.045, "×: a seed failed numerically at that learning rate.", 25)
    fade_overlay(fig, scene_fade(t, t0, t1))


def draw_end(fig, t, t0, t1):
    ftext(fig, 0.5, 0.66, "Shift, not widening", 56)
    lines = ["Smaller proximal steps move the usable learning-rate range up by half a decade to two decades.",
             "ProxProp widens the range in only 3 of 9 settings.",
             "Applying the proximal step to the output layer too (ProxProp-all) widens it in 8 of 9 settings,",
             "removes every numerical failure and raises test accuracy on HAR by 1.1–1.9 points."]
    for k, s in enumerate(lines):
        ftext(fig, 0.5, 0.53 - k * 0.055, s, 25)
    ftext(fig, 0.5, 0.25, "fantasy-creator.github.io/Learning-Rate-Sensitivity-of-Backpropagation", 22, color=MUTED)
    fade_overlay(fig, scene_fade(t, t0, t1, 0.6))


def render(t, path):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    for kind, t0, t1, arg in SCENES:
        if t0 <= t < t1 or (kind == "end" and t >= t0):
            {"title": lambda: draw_title(fig, t, t0, t1), "race": lambda: draw_race(fig, t, t0, t1, arg),
             "sweep": lambda: draw_sweep(fig, t, t0, t1), "end": lambda: draw_end(fig, t, t0, t1)}[kind]()
            break
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)


def work(args):
    frames, folder = args
    load()
    for f in frames:
        render(f / FPS, os.path.join(folder, f"f{f:05d}.png"))
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "assets", "experiments.mp4"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--at", default="", help="comma-separated seconds: write preview PNGs instead of the video")
    args = ap.parse_args()
    if args.at:
        load()
        for s in args.at.split(","):
            render(float(s), args.out.replace(".mp4", "") + f"_{s}.png")
        print("previews written")
        return
    folder = tempfile.mkdtemp(prefix="video-frames-")
    n = int(round(DURATION * FPS))
    chunks = [(list(range(k, n, args.workers)), folder) for k in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        done = sum(pool.map(work, chunks))
    print(f"rendered {done} frames")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(folder, "f%05d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-threads", os.environ.get("FFMPEG_THREADS", "2"), args.out], check=True)
    shutil.rmtree(folder, ignore_errors=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
