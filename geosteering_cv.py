"""
The measuring instrument. 

Output:
  * discover_wells        -> load every well in a directory
  * group_kfold / spatial_kfold -> leak-free folds (by well, or by geographic cluster)
  * randomized_ps_view    -> simulate an earlier Prediction Start (test-like conditions / augmentation)
  * run_cv                -> POOLED RMSE (the leaderboard proxy) + per-well distribution
  * make_submission       -> loop all test wells -> submission.csv (sample order, gap-filled)

Predictors follow a tiny protocol so your transformer drops in later unchanged:
    class P:  def fit(self, train_wells): ...    def predict(self, w) -> full_length_array

.
"""
import glob, os, numpy as np
from geosteering_starter import load_well, baseline_carry, viterbi_correlation, _base_array

# discovery

def discover_wells(root):
    stems = []
    for hp in sorted(glob.glob(f"{root}/*__horizontal_well.csv")):
        stem = hp[:-len("__horizontal_well.csv")]
        if os.path.exists(f"{stem}__typewell.csv"):
            stems.append(stem)
    return [load_well(s) for s in stems]

def _wid(w):  # short well id
    return os.path.basename(w["stem"])

def well_centroid(w):
    """Rough XY centroid from cumulative trajectory increments, for spatial clustering.
    """
    f = w["feats"]
    if not isinstance(f, np.ndarray) or f.shape[1] < 4:
        raise TypeError("well_centroid expects the starter 7-col numpy feats; got "
                        f"{type(f).__name__} with shape {getattr(f,'shape',None)}. "
                        "spatial_kfold needs discover_wells()/load_well() from geosteering_starter.")
    return np.array([np.nancumsum(f[:, 2])[-1], np.nancumsum(f[:, 3])[-1]])


# folds
def group_kfold(well_ids, k=5, seed=0):
    """Whole wells go to exactly one fold. No row of a well ever spans folds."""
    rng = np.random.default_rng(seed)
    ids = list(well_ids); rng.shuffle(ids)
    folds = [ids[i::k] for i in range(k)]           # round-robin split
    return [([x for f2 in folds if f2 is not f for x in f2], f) for f in folds]

def spatial_kfold(wells, k=5, seed=0):
    """Hold out geographic clusters .
    """
    cen = np.array([well_centroid(w) for w in wells])
    ids = [_wid(w) for w in wells]
    if len(wells) < k:                               # too few wells: 1-well folds
        return group_kfold(ids, k=len(wells), seed=seed)
    # simple k-means on centroids
    rng = np.random.default_rng(seed)
    C = cen[rng.choice(len(cen), k, replace=False)]
    for _ in range(25):
        lab = np.argmin(((cen[:,None]-C[None])**2).sum(-1), axis=1)
        for j in range(k):
            if (lab==j).any(): C[j] = cen[lab==j].mean(0)
    folds = [[ids[i] for i in range(len(ids)) if lab[i]==j] for j in range(k)]
    return [([x for f2 in folds if f2 is not f for x in f2], f) for f in folds if f]


# randomized-PS simulation (test-like conditions & augmentation)

def randomized_ps_view(w, rng, frac_lo=0.4, frac_hi=1.0):
    """Return a shallow copy with an EARLIER simulated PS (<= real PS so the anchor
    TVT_input[ps-1] is always a genuinely-known value). eval zone extends from simPS.
    Ground-truth w['tvt'] is left intact for scoring."""
    real_ps = w["ps"]
    lo = max(5, int(frac_lo*real_ps)); hi = max(lo+1, int(frac_hi*real_ps))
    sim = int(rng.integers(lo, hi+1))
    v = dict(w)                                      # shallow copy
    ti = w["tvt_input"].copy(); ti[sim:] = np.nan    # hide TVT after simPS
    em = np.zeros(w["n"], bool); em[sim:] = True
    v["tvt_input"] = ti; v["eval_mask"] = em; v["ps"] = sim
    return v

def real_ps_view(w):
    return w                                         # score at the true PS (LB proxy)


# scoring
def _errs(w, pred):
    if w["tvt"] is None: return np.array([])
    m = w["eval_mask"] & ~np.isnan(w["tvt"]) & ~np.isnan(pred)
    return pred[m] - w["tvt"][m]

def _miss_frac(w):
    m = w["eval_mask"]
    return float(w["feats"][m,1].mean()) if m.any() else 0.0   # col 1 = GR-missing flag


# predictors (tiny protocol; transformer will implement the same two methods)
class CarryPredictor:
    name = "carry"
    def fit(self, wells): return self
    def predict(self, w): return baseline_carry(w)

class ViterbiPredictor:
    name = "viterbi"
    def __init__(self, **kw): self.kw = kw
    def fit(self, wells): return self
    def predict(self, w): return viterbi_correlation(w, **self.kw)


# the harness

def run_cv(root, predictors, k=5, mode="real", seed=0, n_random=2, fold_kind="group"):
    wells = discover_wells(root)
    if not wells:
        print(f"no wells found in {root}"); return
    by_id = {_wid(w): w for w in wells}
    folds = (spatial_kfold(wells, k, seed) if fold_kind=="spatial"
             else group_kfold(list(by_id), k, seed))
    rng = np.random.default_rng(seed)
    print(f"\n=== CV on {root} | {len(wells)} wells | {fold_kind}-{len(folds)}fold | PS mode='{mode}' ===")
    for P in predictors:
        pooled, per_well = [], []
        for tr_ids, va_ids in folds:
            model = P(); model.fit([by_id[i] for i in tr_ids])
            for i in va_ids:
                w = by_id[i]
                views = ([real_ps_view(w)] if mode=="real"
                         else [randomized_ps_view(w, rng) for _ in range(n_random)])
                e_all = []
                for v in views:
                    e = _errs(v, model.predict(v)); e_all.append(e)
                    if len(e): per_well.append((i, float(np.sqrt((e**2).mean())),
                                                int(v["eval_mask"].sum()), _miss_frac(v)))
                e_all = np.concatenate(e_all) if e_all else np.array([])
                if len(e_all): pooled.append(e_all)
        pooled = np.concatenate(pooled) if pooled else np.array([np.nan])
        pw = np.array([r[1] for r in per_well]) if per_well else np.array([np.nan])
        name = getattr(P, "name", P.__name__)
        print(f"\n  [{name}]  POOLED RMSE = {np.sqrt(np.nanmean(pooled**2)):.3f} ft   "
              f"(this tracks the public LB)")
        print(f"     per-well RMSE: mean={np.nanmean(pw):.2f}  median={np.nanmedian(pw):.2f}  "
              f"p90={np.nanpercentile(pw,90):.2f}  worst={np.nanmax(pw):.2f}")
   
        if per_well and len(per_well) > 3:
            mf = np.array([r[3] for r in per_well]); rr = np.array([r[1] for r in per_well])
            if np.std(mf) > 1e-6:
                print(f"     corr(GR-missing-fraction, RMSE) = {np.corrcoef(mf, rr)[0,1]:+.2f}")
        for wid, r, n, mf in sorted(per_well, key=lambda x:-x[1])[:6]:
            print(f"       {wid}: RMSE={r:6.2f}  n={n:5d}  GR-missing={mf*100:3.0f}%")

def make_submission(root, predictor, sample_path, out="submission.csv"):
    """Run predictor over every well in `root`, write competition-format submission."""
    import pandas as pd
    wells = discover_wells(root); model = predictor(); model.fit(wells)
    preds = {_wid(w): model.predict(w) for w in wells}
    ss = pd.read_csv(sample_path)
    def look(r):
        wid, ridx = r["id"].rsplit("_", 1); a = preds.get(wid)
        return float(a[int(ridx)]) if a is not None and int(ridx) < len(a) else np.nan
    ss["tvt"] = ss.apply(look, axis=1)
    n_missing = int(ss["tvt"].isna().sum())
    ss["tvt"] = ss["tvt"].fillna(0.0)                 # safe fallback for any absent well
    ss.to_csv(out, index=False)
    print(f"wrote {out}: {len(ss)} rows ({n_missing} filled by fallback)")

if __name__ == "__main__":
    ROOT = "train"                              
    preds = [CarryPredictor, ViterbiPredictor]
    run_cv(ROOT, preds, k=5, mode="real",   fold_kind="group")
    run_cv(ROOT, preds, k=5, mode="random", fold_kind="group", n_random=3)
    make_submission(ROOT, CarryPredictor, f"{ROOT}/sample_submission.csv",
                    out="submission.csv")
