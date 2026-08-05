"""
EDA — local version.
Run:  python eda.py            (defaults to ROOT="train")
      python eda.py <path>     (custom train dir)

Outputs (into the current folder):
  * eda_summary.csv         -- per-well table (band width, GR-missing %, carry RMSE, ...)
  * eda_1_tvt_vs_md.png ... eda_6_carry_err.png   -- six SEPARATE panels (report-ready)
Also prints the hardest wells (highest carry RMSE) with their band width.
"""
import sys, os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "train"
RESULTS_DIR = "results"    # eda_summary.csv goes here
FIGURES_DIR = "figures"    # all panel PNGs go here
SUB  = 5   # subsample factor for scatter panels

# ----- helpers
def gap_runs(isnan):
    out, c = [], 0
    for v in isnan:
        if v: c += 1
        elif c: out.append(c); c = 0
    if c: out.append(c)
    return np.array(out) if out else np.array([0])

def find_stems(root):
    """Find wells whether files are flat in root/ or nested one level down."""
    hits = glob.glob(f"{root}/*__horizontal_well.csv") or \
           glob.glob(f"{root}/**/*__horizontal_well.csv", recursive=True)
    stems = [hp[:-len("__horizontal_well.csv")] for hp in sorted(hits)
             if os.path.exists(hp[:-len("__horizontal_well.csv")] + "__typewell.csv")]
    return stems

def summarize(stem):
    h = pd.read_csv(f"{stem}__horizontal_well.csv"); t = pd.read_csv(f"{stem}__typewell.csv")
    n = len(h); ps = int(h["TVT_input"].isna().idxmax()); ev = slice(ps, n)
    X, Y = h["X"].values, h["Y"].values
    head = np.degrees(np.arctan2(Y[-1]-Y[0], X[-1]-X[0]))
    tvt = h["TVT"].values if "TVT" in h.columns else np.full(n, np.nan)
    gapn = gap_runs(h["GR"].isna().values)
    carry = np.sqrt(np.nanmean((tvt[ev] - h["TVT_input"].values[ps-1])**2))
    return dict(
        well=os.path.basename(stem), n=n, ps=ps, ps_frac=round(ps/n, 2), eval_n=n-ps,
        lateral_ft=round(float((X[-1]-X[0])**2 + (Y[-1]-Y[0])**2)**0.5), azimuth=round(head, 1),
        tvt_eval_std=round(float(np.nanstd(tvt[ev])), 2),
        tvt_eval_range=round(float(np.nanmax(tvt[ev]) - np.nanmin(tvt[ev])), 1),
        gr_miss_pct=round(100*h["GR"].isna().mean(), 1),
        gr_miss_eval_pct=round(100*np.isnan(h["GR"].values[ev]).mean(), 1),
        gr_gap_med=int(np.median(gapn)), gr_gap_max=int(gapn.max()),
        tw_covers_eval=bool(t["TVT"].min() <= np.nanmin(tvt[ev]) and t["TVT"].max() >= np.nanmax(tvt[ev])),
        carry_rmse=round(float(carry), 2),
    )

#  ----- separate panels
def _fig(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 5)); ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax

def make_panels(stems):
    P = {
        "1_tvt_vs_md":   _fig("TVT vs MD — mean-reverting band", "MD (ft)", "TVT (ft)"),
        "2_dtvt_hist":   _fig("dTVT per ft (eval zone) — increment target", "ΔTVT (ft/ft)", "density"),
        "3_gap_lengths": _fig("GR gap-length distribution — mostly short", "gap length (ft)", "count (log)"),
        "4_gr_match":    _fig("Horizontal GR vs Typewell GR @ matched TVT", "typewell GR", "horizontal GR"),
        "5_tvt_vs_z":    _fig("TVT vs Z — decoupled (why geometry fails)", "Z depth (ft)", "TVT (ft)"),
        "6_carry_err":   _fig("Carry-forward |error| vs distance past PS", "ft past PS", "|error| (ft)"),
    }
    P["1_tvt_vs_md"][1].invert_yaxis()
    for stem in stems:
        try:
            h = pd.read_csv(f"{stem}__horizontal_well.csv"); t = pd.read_csv(f"{stem}__typewell.csv")
        except Exception:
            continue
        ps = int(h["TVT_input"].isna().idxmax()); tvt = h["TVT"].values
        md = h["MD"].values; Z = h["Z"].values
        P["1_tvt_vs_md"][1].plot(md, tvt, lw=.3, alpha=.25)
        d = np.diff(tvt[ps:]); d = d[np.abs(d) < 5]
        P["2_dtvt_hist"][1].hist(d, bins=60, histtype="step", alpha=.2, density=True)
        g = gap_runs(h["GR"].isna().values); g = g[g > 0]
        P["3_gap_lengths"][1].hist(g, bins=np.arange(1, 22), histtype="step", alpha=.2, log=True)
        kn = (~h["TVT_input"].isna().values) & (~h["GR"].isna().values)
        twg = np.interp(tvt[kn], t["TVT"].values, t["GR"].values)
        P["4_gr_match"][1].scatter(twg[::SUB], h["GR"].values[kn][::SUB], s=1, alpha=.03)
        P["5_tvt_vs_z"][1].scatter(Z[ps::SUB], tvt[ps::SUB], s=1, alpha=.03)
        P["6_carry_err"][1].plot((md[ps:]-md[ps]), np.abs(tvt[ps:]-h["TVT_input"].values[ps-1]), lw=.3, alpha=.2)
    for name, (fig, ax) in P.items():
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"eda_{name}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {os.path.join(FIGURES_DIR, f'eda_{name}.png')}")


if __name__ == "__main__":
    stems = find_stems(ROOT)
    print(f"found {len(stems)} wells in '{ROOT}'")
    if not stems:
        sys.exit("No wells found — check the ROOT path and that files end in '__horizontal_well.csv'.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.DataFrame([summarize(s) for s in stems])
    summary_path = os.path.join(RESULTS_DIR, "eda_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    print(f"\nAggregate: wells={len(df)} | median carry RMSE={df.carry_rmse.median():.2f} | "
          f"mean={df.carry_rmse.mean():.2f} | p90={df.carry_rmse.quantile(.9):.2f} | "
          f"typewell covers eval: {df.tw_covers_eval.mean()*100:.0f}%")

    print("\nHARDEST wells (highest carry RMSE) — the tail you must fix:")
    cols = ["well", "carry_rmse", "tvt_eval_std", "tvt_eval_range", "gr_miss_eval_pct", "eval_n"]
    print(df.sort_values("carry_rmse", ascending=False)[cols].head(10).to_string(index=False))

    print("\nWhat predicts a hard well? (correlation of each feature with carry RMSE)")
    for c in ["tvt_eval_std", "tvt_eval_range", "gr_miss_eval_pct", "lateral_ft", "eval_n"]:
        r = df[c].corr(df["carry_rmse"])
        print(f"  corr(carry_rmse, {c:16s}) = {r:+.2f}")

    make_panels(stems)
    print("\ndone.")