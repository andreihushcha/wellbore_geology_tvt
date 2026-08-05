"""
Feature layer + the 'gate each feature on CV' machinery.

"""
import sys, os, glob, numpy as np, pandas as pd
from preprocess import preprocess_well, MODEL_FEATURES
from geosteering_cv import group_kfold, randomized_ps_view, real_ps_view
from geosteering_starter import baseline_carry, viterbi_correlation


def _id(w):
    return w["stem"].split("/")[-1]


#  corr source
def viterbi_corr(w, max_step_ft=1.5, step_pen=0.6, prior_pen=0.15, dgrid=1.0):
    """Global-optimum (Viterbi DP) decode of TVT from the GR<->typewell match,
    anchored at the well's real PS. Returns (corr_tvt, corr_conf), both length n,
    with the pre-PS region left as tvt_input (never used for training; see FIX 2).
    """
    gr   = np.nan_to_num(np.asarray(w["feats"]["gr_norm"]))
    miss = np.asarray(w["feats"]["gr_missing"])
    ps, n = w["ps"], w["n"]
    tw_tvt, tw_gr = w["tw_tvt"], w["tw_gr"]; dg = dgrid
    grid = np.arange(tw_tvt.min(), tw_tvt.max(), dg, dtype=np.float32)
    ref  = np.interp(grid, tw_tvt, tw_gr).astype(np.float32)
    S = len(grid); k = max(1, int(round(max_step_ft / dg))); idx = np.arange(S)
    start = float(w["tvt_input"][ps - 1])
    prior = prior_pen * np.abs(grid - start)
    cost  = np.abs(grid - start).astype(np.float32)
    back  = np.empty((n - ps, S), np.int32)
    conf  = np.zeros(n, np.float32)
    for t in range(ps, n):
        best = np.full(S, np.inf, np.float32); arg = idx.copy()
        for o in range(-k, k + 1):
            src = np.clip(idx - o, 0, S - 1)
            c = cost[src] + step_pen * abs(o) * dg
            b = c < best; best[b] = c[b]; arg[b] = src[b]
        cost = best + (1 - miss[t]) * np.abs(gr[t] - ref) + prior
        back[t - ps] = arg
        # confidence = how sharply this step separates the winner from the field
        conf[t] = float((np.median(cost) - cost.min()) / (cost.std() + 1e-6))
    j = int(np.argmin(cost)); path_idx = np.empty(n - ps, np.int32)
    for t in range(n - ps - 1, -1, -1):
        path_idx[t] = j; j = int(back[t, j])
    corr_tvt = w["tvt_input"].astype(np.float32).copy()
    corr_tvt[ps:] = grid[path_idx]
    return corr_tvt, conf


def windowed_correlation(w, win=25, max_step_ft=1.5, dgrid=0.5,
                         stretches=(-0.8,-0.5,-0.3,-0.15,0.15,0.3,0.5,0.8), sub=3):
    """DEPRECATED as a feature source (greedy, drifts). Kept for reference only.
    Use viterbi_corr for the `corr` feature."""
    gr = np.asarray(w["feats"]["gr_norm"]) if isinstance(w["feats"], pd.DataFrame) else w["feats"][:,0]
    ps, n = w["ps"], w["n"]; tw_tvt, tw_gr = w["tw_tvt"], w["tw_gr"]
    anchor = float(w["tvt_input"][ps-1]); off = np.arange(-(win//2), win//2+1)
    step = max_step_ft * sub
    corr_tvt = w["tvt_input"].copy(); corr_conf = np.zeros(n, np.float32); prev = anchor
    for t in range(ps, n, sub):
        lo, hi = max(0, t-win//2), min(n, t+win//2+1); lw = gr[lo:hi]
        cand = np.arange(prev-step, prev+step+dgrid, dgrid)
        cand = cand[(cand > tw_tvt.min()+win*dgrid) & (cand < tw_tvt.max()-win*dgrid)]
        if len(lw) < win or np.std(lw) < 1e-6 or len(cand) == 0:
            corr_tvt[t] = prev; continue
        lw = (lw - lw.mean()) / (lw.std() + 1e-6)
        best_cost, best_c, allc = np.inf, prev, []
        for s in stretches:
            samp = cand[:, None] + (s * off)[None, :] * dgrid
            tw_win = np.interp(samp.ravel(), tw_tvt, tw_gr).reshape(samp.shape)
            tw_win = (tw_win - tw_win.mean(1, keepdims=True)) / (tw_win.std(1, keepdims=True) + 1e-6)
            cost = 1 - (tw_win * lw[None, :]).mean(1); allc.append(cost)
            j = int(np.argmin(cost))
            if cost[j] < best_cost: best_cost, best_c = cost[j], cand[j]
        allc = np.concatenate(allc); prev = best_c; corr_tvt[t] = best_c
        corr_conf[t] = float((np.median(allc) - best_cost) / (allc.std() + 1e-6))
    filled = pd.Series(corr_tvt).interpolate(limit_area="inside").values
    corr_tvt = np.where(np.isnan(filled), anchor, filled)
    corr_conf = pd.Series(corr_conf).interpolate(limit_area="inside").fillna(0).values
    return corr_tvt, corr_conf


def cache_corr(W, **vk):
    """Cache the Viterbi corr decode once per well ."""
    for i, w in enumerate(W):
        w["corr_tvt"], w["corr_conf"] = viterbi_corr(w, **vk)
        if (i + 1) % 25 == 0:
            print(f"  cached corr {i+1}/{len(W)}")
    return W


# predictors
class CarryP:
    name = "carry"
    def fit(self, W): return self
    def predict(self, w):
        p = w["tvt_input"].copy(); p[w["eval_mask"]] = w["tvt_input"][w["ps"]-1]; return p


class WindowedCorrP:
    name = "viterbi_corr(seq)"
    def fit(self, W): return self
    def predict(self, w):
        corr = w.get("corr_tvt")
        if corr is None: corr = viterbi_corr(w)[0]          # parity: Viterbi, not greedy
        out = w["tvt_input"].copy(); out[w["eval_mask"]] = corr[w["eval_mask"]]; return out


class ViterbiP:
    name = "viterbi(pointwise)"
    def __init__(self, max_step_ft=1.5, step_pen=0.6, prior_pen=0.15, dgrid=1.0):
        self.__dict__.update(max_step_ft=max_step_ft, step_pen=step_pen, prior_pen=prior_pen, dgrid=dgrid)
        self.name = f"viterbi(pp={prior_pen},sp={step_pen})"
    def fit(self, W): return self
    def predict(self, w):
        gr = np.nan_to_num(np.asarray(w["feats"]["gr_norm"])); miss = np.asarray(w["feats"]["gr_missing"])
        ps, n = w["ps"], w["n"]; tw_tvt, tw_gr = w["tw_tvt"], w["tw_gr"]; dg = self.dgrid
        grid = np.arange(tw_tvt.min(), tw_tvt.max(), dg, dtype=np.float32)
        ref = np.interp(grid, tw_tvt, tw_gr).astype(np.float32)
        S = len(grid); k = max(1, int(round(self.max_step_ft/dg))); idx = np.arange(S)
        start = float(w["tvt_input"][ps-1])
        prior = self.prior_pen*np.abs(grid-start); cost = np.abs(grid-start).astype(np.float32)
        back = np.empty((n-ps, S), np.int32)
        for t in range(ps, n):
            best = np.full(S, np.inf, np.float32); arg = idx.copy()
            for o in range(-k, k+1):
                src = np.clip(idx-o, 0, S-1); c = cost[src]+self.step_pen*abs(o)*dg
                b = c < best; best[b] = c[b]; arg[b] = src[b]
            cost = best + (1-miss[t])*np.abs(gr[t]-ref) + prior; back[t-ps] = arg
        j = int(np.argmin(cost)); path = np.empty(n-ps, np.float32)
        for t in range(n-ps-1, -1, -1): path[t] = grid[j]; j = int(back[t, j])
        out = w["tvt_input"].copy(); out[ps:] = path; return out


class GBMPredictor:
    GROUPS = {
        "gr":   ["gr_norm","gr_d1","gr_d2","gr_roll_mean","gr_roll_std","gr_roll_range"],
        "miss": ["gr_missing","gr_interp","gr_missfrac"],
        "geom": ["dZ","incl","azim_sin","azim_cos","dogleg","vs_dist_norm","md_pos"],
        "band": ["dist_from_ps_norm","known_flag"],
        "corr": ["corr_tvt_rel","corr_conf"],
    }
    def __init__(self, groups=("gr","miss","geom","band"), use_corr=False):
        self.groups = list(groups) + (["corr"] if use_corr else [])
        self.use_corr = use_corr
        self.name = "gbm[" + "+".join(self.groups) + "]"
    def _cols(self):
        c = []; [c.extend(self.GROUPS[g]) for g in self.groups]; return c
    def _X(self, w):
        f = w["feats"].copy()
        if self.use_corr:
            corr = w.get("corr_tvt"); conf = w.get("corr_conf")
            if corr is None: corr, conf = viterbi_corr(w)          # parity: Viterbi, not greedy
            f["corr_tvt_rel"] = corr - w["tvt_input"][w["ps"]-1]; f["corr_conf"] = conf
        return f[self._cols()].values

    @staticmethod
    def _train_rows(w):
        """FIX 2 -- parity: train only on the from-PS region, and only where the
        target is known. Excludes pre-PS rows where corr_tvt == tvt_input == label."""
        rows = np.zeros(w["n"], bool); rows[w["ps"]:] = True
        rows &= ~np.isnan(w["tvt"])
        return rows

    def fit(self, W):
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xs, ys = [], []
        for w in W:
            if w["tvt"] is None: continue
            rows = self._train_rows(w)
            if not rows.any(): continue
            X = self._X(w); anchor = w["tvt_input"][w["ps"]-1]
            Xs.append(X[rows]); ys.append(w["tvt"][rows] - anchor)
        self.m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08,
                                               max_iter=300, l2_regularization=1.0)
        self.m.fit(np.vstack(Xs), np.concatenate(ys)); return self
    def predict(self, w):
        anchor = w["tvt_input"][w["ps"]-1]
        pred = self.m.predict(self._X(w)) + anchor
        out = w["tvt_input"].copy(); out[w["eval_mask"]] = pred[w["eval_mask"]]; return out


# CV core
def _pooled(W, folds, make, mode, rng, n_random, per_well=False):
    err = []; pw = {}
    for tr_ids, va_ids in folds:
        model = make(); model.fit([w for w in W if _id(w) in tr_ids])
        for w in W:
            if _id(w) not in va_ids: continue
            views = [real_ps_view(w)] if mode=="real" else [randomized_ps_view(w, rng) for _ in range(n_random)]
            we = []
            for v in views:
                p = model.predict(v); m = v["eval_mask"] & ~np.isnan(v["tvt"]) & ~np.isnan(p)
                d = p[m] - v["tvt"][m]; err.append(d); we.append(d)
            if per_well and we:
                we = np.concatenate(we)
                if len(we): pw[_id(w)] = (float(np.sqrt(np.mean(we**2))), len(we))
    e = np.concatenate(err) if err else np.array([np.nan])
    pooled = float(np.sqrt(np.nanmean(e**2)))
    return (pooled, pw) if per_well else pooled


def _print_worst(pw, tag, k=8):
    if not pw: return
    print(f"\n  worst wells [{tag}]:")
    for wid,(r,n) in sorted(pw.items(), key=lambda kv: -kv[1][0])[:k]:
        print(f"       {wid}: RMSE={r:6.2f}  n={n}")


def compare(W, makers, k=5, seed=0, save="results/features_cv.csv", diag_last=2):
    ids = [_id(w) for w in W]
    folds = group_kfold(ids, k=min(k, len(ids)), seed=seed)
    rows = []; prev = None
    print(f"\n{'predictor':34s} {'POOLED real':>12s} {'delta vs prev':>14s}")
    print("-"*62)
    diag_from = max(0, len(makers) - diag_last)
    for i, make in enumerate(makers):
        name = make().name
        out = _pooled(W, folds, make, "real", np.random.default_rng(seed), 1, per_well=(i>=diag_from))
        r, pw = out if isinstance(out, tuple) else (out, None)
        d = "" if prev is None else f"{r-prev:+.3f}"
        print(f"{name:34s} {r:12.3f} {d:>14s}"); rows.append(dict(predictor=name, pooled_real=round(r,3)))
        if pw is not None: _print_worst(pw, name)
        prev = r
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        pd.DataFrame(rows).to_csv(save, index=False); print(f"\nwrote {save}")


def sweep_viterbi(W, k=5, seed=0, priors=(0.05,0.1,0.15,0.25,0.4), steps=(0.3,0.6,1.0)):
    """sweep Viterbi prior_pen/step_pen on the set to fix
    anchor-pull. Prints pooled RMSE per (prior,step) and the worst wells for the
    best combo, so we can see whether the six tail wells collapse."""
    ids = [_id(w) for w in W]; folds = group_kfold(ids, k=min(k,len(ids)), seed=seed)
    print(f"\n{'prior_pen':>9s} {'step_pen':>9s} {'POOLED real':>12s}")
    print("-"*34); best = (np.inf, None, None)
    for pp in priors:
        for sp in steps:
            r = _pooled(W, folds, lambda pp=pp, sp=sp: ViterbiP(prior_pen=pp, step_pen=sp),
                        "real", np.random.default_rng(seed), 1)
            print(f"{pp:9.2f} {sp:9.2f} {r:12.3f}")
            if r < best[0]: best = (r, pp, sp)
    _, pp, sp = best
    print(f"\nbest: prior_pen={pp}, step_pen={sp}  pooled_real={best[0]:.3f}")
    _, pw = _pooled(W, folds, lambda: ViterbiP(prior_pen=pp, step_pen=sp),
                    "real", np.random.default_rng(seed), 1, per_well=True)
    _print_worst(pw, f"viterbi(pp={pp},sp={sp})")
    return best


if __name__ == "__main__":
    ROOT   = sys.argv[1] if len(sys.argv) > 1 else "train"
    SUBSET = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    MODE   = sys.argv[3] if len(sys.argv) > 3 else "ablation"
    stems = [hp[:-len("__horizontal_well.csv")]
             for hp in sorted(glob.glob(f"{ROOT}/*__horizontal_well.csv") or
                              glob.glob(f"{ROOT}/**/*__horizontal_well.csv", recursive=True))
             if os.path.exists(hp[:-len("__horizontal_well.csv")]+"__typewell.csv")]
    if SUBSET and SUBSET < len(stems):
        rng = np.random.default_rng(0); stems = list(rng.choice(stems, SUBSET, replace=False))
    print(f"using {len(stems)} wells (SUBSET={SUBSET or 'ALL'}) from '{ROOT}'")
    print("preprocessing + caching Viterbi correlation (slow part) ...")
    W = cache_corr([preprocess_well(s) for s in stems])

    if MODE == "sweep":
        sweep_viterbi(W)
    else:
        makers = [
            lambda: CarryP(),
            lambda: GBMPredictor(groups=("gr",), use_corr=False),
            lambda: GBMPredictor(groups=("gr","geom"), use_corr=False),
            lambda: GBMPredictor(groups=("gr","geom","band"), use_corr=False),
            lambda: GBMPredictor(groups=("gr","geom","band","miss"), use_corr=False),
            lambda: GBMPredictor(groups=("gr","geom","band","miss"), use_corr=True),
            lambda: ViterbiP(),
        ]
        compare(W, makers, k=5)