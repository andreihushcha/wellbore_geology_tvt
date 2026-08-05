"""
KAGGLE SUBMISSION NOTEBOOK  (self-contained, no external module imports)

"""
import os, glob, numpy as np, pandas as pd

#config
FAST = True          # True: coarse 1.0-ft TVT grid (fast, ~fine). False: 0.5-ft (slower).
DGRID = 1.0 if FAST else 0.5
GAP_MAX_FT = 5.0     # interpolate GR gaps <= this; mask longer ones

#  paths
def find_input_root():
    """Locate the competition data dir under /kaggle/input (or fall back to cwd)."""
    for base in ["/kaggle/input", "."]:
        # a dir that directly contains test/ , or contains *__horizontal_well.csv
        for cand in glob.glob(f"{base}/**/", recursive=True):
            if os.path.isdir(os.path.join(cand, "test")) or glob.glob(f"{cand}/*__horizontal_well.csv"):
                return cand.rstrip("/")
    return "."

ROOT = find_input_root()
TEST_DIR = os.path.join(ROOT, "test") if os.path.isdir(os.path.join(ROOT, "test")) else ROOT
SAMPLE = next(iter(glob.glob(f"{ROOT}/**/sample_submission.csv", recursive=True)), None)
OUT = "/kaggle/working/submission.csv" if os.path.isdir("/kaggle/working") else "submission.csv"
print(f"ROOT={ROOT}\nTEST_DIR={TEST_DIR}\nSAMPLE={SAMPLE}\nOUT={OUT}")

# io + clean
def robust_norm(x):
    x = np.asarray(x, float); med = np.nanmedian(x)
    iqr = (np.nanpercentile(x, 75) - np.nanpercentile(x, 25)) + 1e-6
    return (x - med) / iqr

def _runs(isn):
    i, n = 0, len(isn)
    while i < n:
        if isn[i]:
            j = i
            while j < n and isn[j]: j += 1
            yield i, j - i; i = j
        else: i += 1

def clean_gr(gr, md, gap_max_ft=GAP_MAX_FT):
    gr = np.asarray(gr, float); isn = np.isnan(gr)
    filled = pd.Series(gr).interpolate(method="linear", limit_area="inside").values
    hard = isn.copy()
    for s, L in _runs(isn):
        span = md[min(s+L, len(md)-1)] - md[s]
        if span <= gap_max_ft and not np.isnan(filled[s:s+L]).any(): hard[s:s+L] = False
    filled = np.where(hard, np.nan, filled)
    return filled, hard.astype(np.float32)

def load_well(stem):
    h = pd.read_csv(f"{stem}__horizontal_well.csv"); t = pd.read_csv(f"{stem}__typewell.csv")
    md = h["MD"].values.astype(float); n = len(h)
    ps = int(h["TVT_input"].isna().idxmax()); eval_mask = h["TVT_input"].isna().values
    gr_filled, gr_hard = clean_gr(h["GR"].values, md)
    gr_norm = np.nan_to_num(robust_norm(gr_filled), nan=0.0)
    return dict(stem=stem, n=n, ps=ps, eval_mask=eval_mask, md=md,
                tvt_input=h["TVT_input"].values.astype(float),
                gr=gr_norm, miss=gr_hard,
                tw_tvt=t["TVT"].values.astype(np.float32),
                tw_gr=robust_norm(t["GR"].values).astype(np.float32))

#  predictors
def predict_carry(w):
    p = w["tvt_input"].copy(); p[w["eval_mask"]] = w["tvt_input"][w["ps"]-1]; return p

def predict_viterbi(w, max_step_ft=1.5, step_pen=0.6, prior_pen=0.15, dgrid=DGRID):
    """Banded Viterbi: match GR to typewell GR(TVT), smoothness + mean-reversion prior."""
    gr = w["gr"]; miss = w["miss"]; ps, n = w["ps"], w["n"]
    tw_tvt, tw_gr = w["tw_tvt"], w["tw_gr"]
    grid = np.arange(tw_tvt.min(), tw_tvt.max(), dgrid, dtype=np.float32)
    ref = np.interp(grid, tw_tvt, tw_gr).astype(np.float32)
    S = len(grid); k = max(1, int(round(max_step_ft/dgrid))); idx = np.arange(S)
    start = float(w["tvt_input"][ps-1])
    prior = prior_pen*np.abs(grid-start).astype(np.float32)
    cost = np.abs(grid-start).astype(np.float32); back = np.empty((n-ps, S), np.int32)
    for t in range(ps, n):
        best = np.full(S, np.inf, np.float32); arg = idx.copy()
        for o in range(-k, k+1):
            src = np.clip(idx-o, 0, S-1); c = cost[src]+step_pen*abs(o)*dgrid
            b = c < best; best[b] = c[b]; arg[b] = src[b]
        cost = best + (1-miss[t])*np.abs(gr[t]-ref) + prior; back[t-ps] = arg
    j = int(np.argmin(cost)); path = np.empty(n-ps, np.float32)
    for t in range(n-ps-1, -1, -1): path[t] = grid[j]; j = int(back[t, j])
    out = w["tvt_input"].copy(); out[ps:] = path; return out

PREDICTOR = predict_viterbi    

def main():
    stems = sorted(hp[:-len("__horizontal_well.csv")]
                   for hp in glob.glob(f"{TEST_DIR}/*__horizontal_well.csv")
                   if os.path.exists(hp[:-len("__horizontal_well.csv")] + "__typewell.csv"))
    print(f"discovered {len(stems)} test wells")
    preds = {}
    for i, stem in enumerate(stems):
        wid = os.path.basename(stem)
        try:
            w = load_well(stem)
            preds[wid] = PREDICTOR(w)
        except Exception as e:                       
            print(f"  [warn] {wid}: {e} -> carry fallback")
            try: preds[wid] = predict_carry(load_well(stem))
            except Exception: pass
        if (i+1) % 25 == 0: print(f"  {i+1}/{len(stems)} done")

    # build submission aligned to sample_submission (defines the exact required ids)
    if SAMPLE:
        ss = pd.read_csv(SAMPLE)
        def look(r):
            wid, ridx = r["id"].rsplit("_", 1); a = preds.get(wid)
            return float(a[int(ridx)]) if a is not None and int(ridx) < len(a) else np.nan
        ss["tvt"] = ss.apply(look, axis=1)
        nmiss = int(ss["tvt"].isna().sum()); ss["tvt"] = ss["tvt"].fillna(0.0)
        ss.to_csv(OUT, index=False)
        print(f"wrote {OUT}: {len(ss)} rows, {nmiss} fallback-filled")
    else:                                            # no sample file: emit every eval row
        rows = []                                    # load each well ONCE (not per row)
        for wid, a in preds.items():
            w = load_well(f"{TEST_DIR}/{wid}")
            for i in np.where(w["eval_mask"])[0]:
                rows.append((f"{wid}_{int(i)}", float(a[int(i)])))
        pd.DataFrame(rows, columns=["id", "tvt"]).to_csv(OUT, index=False)
        print(f"wrote {OUT}: {len(rows)} rows")

if __name__ == "__main__":
    main()
