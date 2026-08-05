"""
Preprocessing: turn a raw well into clean model features.

"""
import numpy as np, pandas as pd

MODEL_FEATURES = [                      # columns fed to the model (all NaN-free after clean)
    "gr_norm", "gr_missing", "gr_interp", "gr_missfrac",
    "gr_d1", "gr_d2", "gr_roll_mean", "gr_roll_std", "gr_roll_range",
    "dZ", "incl", "azim_sin", "azim_cos", "dogleg", "vs_dist_norm", "md_pos",
    "known_flag", "dist_from_ps_norm",
]

def _run_lengths(isnan):
    """Yield (start, length) of each contiguous NaN run."""
    i, n = 0, len(isnan)
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]: j += 1
            yield i, j - i; i = j
        else:
            i += 1

def clean_gr(gr, md, gap_max_ft=5.0):
    """Interpolate short gaps, mask long ones. Returns (gr_filled, missing_hard, interp_flag).
    md is used so the threshold is in FEET even if sampling isn't exactly 1 ft."""
    gr = np.asarray(gr, float); isn = np.isnan(gr)
    filled = pd.Series(gr).interpolate(method="linear", limit_area="inside").values
    missing_hard = isn.copy()            # start: everything missing is 'hard'
    interp_flag = np.zeros_like(gr)
    for s, L in _run_lengths(isn):
        span_ft = md[min(s+L, len(md)-1)] - md[s]
        if span_ft <= gap_max_ft and not np.isnan(filled[s:s+L]).any():
            missing_hard[s:s+L] = False  # short interior gap -> keep interpolation
            interp_flag[s:s+L] = 1.0
    # leading/trailing NaN stay hard-missing (interpolate can't fill them)
    filled = np.where(missing_hard, np.nan, filled)
    return filled, missing_hard.astype(np.float32), interp_flag.astype(np.float32)

def robust_norm(x):
    x = np.asarray(x, float); med = np.nanmedian(x)
    iqr = (np.nanpercentile(x, 75) - np.nanpercentile(x, 25)) + 1e-6
    return (x - med) / iqr, med, iqr

def _roll(x, win, fn):
    """NaN-aware centered rolling stat."""
    s = pd.Series(x)
    return s.rolling(win, center=True, min_periods=max(3, win//4)).apply(fn, raw=True).bfill().ffill().values

def preprocess_well(stem, gap_max_ft=5.0, roll_ft=25):
    """Return dict with clean feature DataFrame + arrays. Test-safe (tvt=None if absent)."""
    h = pd.read_csv(f"{stem}__horizontal_well.csv")
    t = pd.read_csv(f"{stem}__typewell.csv")
    md = h["MD"].values.astype(float); n = len(h)
    ps = int(h["TVT_input"].isna().idxmax())
    eval_mask = h["TVT_input"].isna().values

    #  GR cleaning + normalization (per-well, observed-only stats) 
    gr_filled, gr_missing, gr_interp = clean_gr(h["GR"].values, md, gap_max_ft)
    gr_norm, gmed, giqr = robust_norm(gr_filled)
    gr_norm = np.nan_to_num(gr_norm, nan=0.0)           # hard-missing -> 0 (=median); mask flags it
    gr_missfrac = _roll(gr_missing, roll_ft, np.mean)
    gr_d1 = np.gradient(gr_norm)
    gr_d2 = np.gradient(gr_d1)
    gr_roll_mean = _roll(gr_norm, roll_ft, np.mean)
    gr_roll_std  = _roll(gr_norm, roll_ft, np.std)
    gr_roll_range = _roll(gr_norm, roll_ft, lambda a: np.nanmax(a) - np.nanmin(a))

    # geometry (all in feet)
    X, Y, Z = h["X"].values, h["Y"].values, h["Z"].values
    dX, dY, dZ = np.gradient(X), np.gradient(Y), np.gradient(Z)
    incl = np.arctan2(np.hypot(dX, dY), -dZ + 1e-9)      # from vertical: ~pi/2 when horizontal
    azim = np.arctan2(dY, dX)
    heading = np.arctan2(Y[-1]-Y[0], X[-1]-X[0])         # net azimuth
    ux, uy = np.cos(heading), np.sin(heading)
    vs = (X-X[0])*ux + (Y-Y[0])*uy                       # vector-section distance (ft)
    vs_norm = (vs - vs.min()) / (vs.max() - vs.min() + 1e-9)
    dogleg = np.gradient(np.unwrap(azim))
    md_pos = (md - md[0]) / (md[-1] - md[0] + 1e-9)

    # known/anchor context (bounds the mean-reverting band)
    tvt_input = h["TVT_input"].values.astype(float)
    known_flag = (~np.isnan(tvt_input)).astype(np.float32)
    anchor = float(tvt_input[ps-1])
    kt = tvt_input[:ps]
    known_stats = dict(anchor=anchor, kmean=np.nanmean(kt), kstd=np.nanstd(kt),
                       kmin=np.nanmin(kt), kmax=np.nanmax(kt))
    dist_from_ps = (np.arange(n) - ps).astype(float)
    dist_from_ps_norm = dist_from_ps / (n - ps + 1e-9)

    feats = pd.DataFrame({
        "gr_norm": gr_norm, "gr_missing": gr_missing, "gr_interp": gr_interp, "gr_missfrac": gr_missfrac,
        "gr_d1": gr_d1, "gr_d2": gr_d2, "gr_roll_mean": gr_roll_mean,
        "gr_roll_std": gr_roll_std, "gr_roll_range": gr_roll_range,
        "dZ": dZ, "incl": incl, "azim_sin": np.sin(azim), "azim_cos": np.cos(azim),
        "dogleg": dogleg, "vs_dist_norm": vs_norm, "md_pos": md_pos,
        "known_flag": known_flag, "dist_from_ps_norm": dist_from_ps_norm,
    })[MODEL_FEATURES]

    tvt = h["TVT"].values.astype(float) if "TVT" in h.columns else None
    tw_gr_norm, _, _ = robust_norm(t["GR"].values)
    return dict(stem=stem, feats=feats, md=md, ps=ps, eval_mask=eval_mask, n=n,
                tvt=tvt, tvt_input=tvt_input, gr_norm_stats=(gmed, giqr),
                tw_tvt=t["TVT"].values.astype(np.float32), tw_gr=tw_gr_norm.astype(np.float32),
                known=known_stats)

def _report(stem):
    raw = pd.read_csv(f"{stem}__horizontal_well.csv")
    w = preprocess_well(stem)
    f = w["feats"]
    n_nan = int(f.isna().sum().sum())
    gr_raw_nan = int(raw["GR"].isna().sum())
    interp = int(w["feats"]["gr_interp"].sum())
    hard = int(w["feats"]["gr_missing"].sum())
    wid = stem.split("/")[-1]
    print(f"{wid}: features={f.shape[1]} cols x {f.shape[0]} rows | NaNs in model features={n_nan} "
          f"| GR raw-NaN={gr_raw_nan} -> interpolated={interp}, masked-hard={hard} "
          f"| target={'present' if w['tvt'] is not None else 'ABSENT (test)'}")
    return w

if __name__ == "__main__":
    import glob, os
    for root in ["train"]:
        for hp in sorted(glob.glob(f"{root}/*__horizontal_well.csv")):
            stem = hp[:-len("__horizontal_well.csv")]
            if os.path.exists(f"{stem}__typewell.csv"): _report(stem)
