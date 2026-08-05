"""
Geosteering TVT prediction
"""
import numpy as np, pandas as pd
from pathlib import Path

# never feed these to the model — training-only, absent at test time
FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
GEOLOGY_VOCAB = ["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","LTHL","LTGT","LBHL","MNSS","BUDA"]


# 1. DATA

def robust_norm(x):
    x = np.asarray(x, float); med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25) + 1e-6
    return (x - med) / iqr, med, iqr

def load_well(stem):
    """stem = path prefix, e.g. '/data/test/000d7d20'. Works for BOTH schemas:
       train: MD,X,Y,Z,ANCC..BUDA,TVT,GR,TVT_input   (has ground truth)
       test : MD,X,Y,Z,GR,TVT_input                  (no TVT, no formation tops)
    Returns dict; w['tvt'] is None when ground truth is absent."""
    h = pd.read_csv(f"{stem}__horizontal_well.csv")
    t = pd.read_csv(f"{stem}__typewell.csv")
    ps = int(h["TVT_input"].isna().idxmax())          # first eval row
    eval_mask = h["TVT_input"].isna().values           # rows to predict
    gr = h["GR"].values.astype(float)
    gr_missing = np.isnan(gr).astype(np.float32)        # ~half of eval GR is NaN here
    gr_norm, gmed, giqr = robust_norm(gr)
    gr_norm = np.nan_to_num(gr_norm, nan=0.0)
    dX = np.gradient(h["X"].values); dY = np.gradient(h["Y"].values); dZ = np.gradient(h["Z"].values)
    incl = np.arctan2(np.hypot(dX, dY), -dZ + 1e-9)
    md = h["MD"].values; md_pos = (md - md[0]) / (md[-1] - md[0] + 1e-9)
    feats = np.stack([gr_norm, gr_missing, dX, dY, dZ, incl, md_pos], axis=1).astype(np.float32)
    tw_gr = ((t["GR"].values - gmed) / giqr).astype(np.float32)
    tw_tvt = t["TVT"].values.astype(np.float32)
    if "Geology" in t.columns:                          # Geology is train-only in test files
        geo_idx = t["Geology"].map({g:i for i,g in enumerate(GEOLOGY_VOCAB)}).fillna(-1).astype(int).values
    else:
        geo_idx = np.full(len(t), -1, dtype=int)
    tvt = h["TVT"].values.astype(float) if "TVT" in h.columns else None   # None on test
    return dict(stem=stem, feats=feats, md=md, ps=ps, eval_mask=eval_mask,
                tvt=tvt, tvt_input=h["TVT_input"].values.astype(float),
                gr=gr, tw_gr=tw_gr, tw_tvt=tw_tvt, geo_idx=geo_idx, n=len(h))


# 2. BASELINES

def _base_array(w):
    """Full-length array seeded with known TVT_input; eval zone filled by a predictor."""
    a = w["tvt_input"].copy()
    a[w["eval_mask"]] = np.nan
    return a

def baseline_carry(w):
    pred = _base_array(w)
    pred[w["eval_mask"]] = w["tvt_input"][w["ps"]-1]     # last known TVT (the anchor)
    return pred

def viterbi_correlation(w, max_step_ft=1.5, step_pen=0.6, prior_pen=0.15, dgrid=0.5):
    """Smoothest TVT path matching lateral GR to typewell GR(TVT), with a mean-reversion
    prior toward the PS anchor (the driller steers back to zone). Banded => O(N*S*k).
    """
    tw_tvt, tw_gr = w["tw_tvt"], w["tw_gr"]
    gr = np.nan_to_num(w["feats"][:,0]); miss = w["feats"][:,1]
    ps, n = w["ps"], w["n"]
    grid = np.arange(tw_tvt.min(), tw_tvt.max(), dgrid, dtype=np.float32)
    tw_on_grid = np.interp(grid, tw_tvt, tw_gr).astype(np.float32)
    S = len(grid); k = int(round(max_step_ft/dgrid))
    start = float(w["tvt_input"][ps-1])
    prior = prior_pen*np.abs(grid - start).astype(np.float32)   # pull toward zone
    cost = np.abs(grid - start).astype(np.float32)
    back = np.empty((n-ps, S), np.int32); idx = np.arange(S)
    for t in range(ps, n):
        best = np.full(S, np.inf, np.float32); arg = idx.copy()
        for off in range(-k, k+1):
            src = np.clip(idx - off, 0, S-1)
            cand = cost[src] + step_pen*abs(off)*dgrid
            better = cand < best; best[better] = cand[better]; arg[better] = src[better]
        emis = (1.0-miss[t])*np.abs(gr[t] - tw_on_grid)  # ignore emission where GR missing
        cost = best + emis + prior
        back[t-ps] = arg
    j = int(np.argmin(cost)); path = np.empty(n-ps, np.float32)
    for t in range(n-ps-1, -1, -1):
        path[t] = grid[j]; j = int(back[t, j])
    pred = _base_array(w); pred[ps:] = path
    return pred

def rmse_eval(w, pred):
    if w["tvt"] is None:
        return float("nan")                              # test well: no ground truth
    m = w["eval_mask"] & ~np.isnan(w["tvt"])
    return float(np.sqrt(np.mean((pred[m]-w["tvt"][m])**2)))


# 3. TRANSFORMER  (dual-encoder, cross-attention as differentiable correlation)

try:
    import torch, torch.nn as nn
    class ConvStem(nn.Module):
        def __init__(s, cin, d):
            super().__init__(); s.net = nn.Sequential(
                nn.Conv1d(cin, d, 5, padding=2), nn.GELU(),
                nn.Conv1d(d, d, 5, padding=2), nn.GELU())
        def forward(s, x): return s.net(x.transpose(1,2)).transpose(1,2)  # B,L,d

    class GeoTransformer(nn.Module):
        """Lateral queries the typewell via cross-attention -> predicts dTVT increments."""
        def __init__(s, n_feat=7, d=256, heads=8, enc=4, dec=4, n_geo=len(GEOLOGY_VOCAB)):
            super().__init__()
            s.lat_stem = ConvStem(n_feat, d)
            s.tw_stem  = ConvStem(1 + n_geo, d)          # tw GR + geology one-hot
            el = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True)
            s.lat_enc = nn.TransformerEncoder(el, enc)   # BIDIRECTIONAL over GR
            s.tw_enc  = nn.TransformerEncoder(nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True), 2)
            dl = nn.TransformerDecoderLayer(d, heads, 4*d, batch_first=True)
            s.dec = nn.TransformerDecoder(dl, dec)       # cross-attn lateral->typewell
            s.tvt_embed = nn.Linear(1, d)                # tag typewell tokens with their TVT
            s.head  = nn.Linear(d, 1)                    # dTVT per step
            s.sigma = nn.Linear(d, 1)                    # optional log-variance
        def forward(s, lat, tw, tw_tvt):
            H = s.lat_enc(s.lat_stem(lat))
            M = s.tw_enc(s.tw_stem(tw)) + s.tvt_embed(tw_tvt.unsqueeze(-1))
            z = s.dec(H, M)                              # attention weights ~ correlation
            return s.head(z).squeeze(-1), s.sigma(z).squeeze(-1)

    def reconstruct_tvt(dtvt, anchor, ps):
        """cumsum increments from the exact known TVT(PS)."""
        out = torch.zeros_like(dtvt); out[:, :ps] = 0
        out[:, ps:] = anchor.unsqueeze(1) + torch.cumsum(dtvt[:, ps:], dim=1)
        return out
    TORCH_OK = True
except Exception as e:  
    TORCH_OK = False
    print("torch not available; baselines still run.", e)


# 4. SUBMISSION

def write_submission(preds_by_well, sample_path, out="submission.csv"):
    """preds_by_well[well] = full-length TVT array. Reindex to sample order, fill gaps."""
    ss = pd.read_csv(sample_path)
    def lookup(row):
        well, ridx = row["id"].rsplit("_", 1); ridx = int(ridx)
        arr = preds_by_well.get(well)
        return float(arr[ridx]) if arr is not None and ridx < len(arr) else 0.0
    ss["tvt"] = ss.apply(lookup, axis=1)
    ss.to_csv(out, index=False); print("wrote", out, len(ss), "rows")

if __name__ == "__main__":
    import glob, os

    roots = ["train"]
    preds_by_well = {}
    for root in roots:
        for hp in sorted(glob.glob(f"{root}/*__horizontal_well.csv")):
            stem = hp.replace("__horizontal_well.csv", "")
            well = os.path.basename(stem)
            if not os.path.exists(f"{stem}__typewell.csv"):
                continue
            w = load_well(stem)
            pred = viterbi_correlation(w)              
            preds_by_well.setdefault(well, pred)
            if w["tvt"] is not None:
                print(f"{well}: train RMSE={rmse_eval(w,pred):.2f}  (carry={rmse_eval(w,baseline_carry(w)):.2f})")
            else:
                print(f"{well}: TEST well (no ground truth) -> {int(w['eval_mask'].sum())} preds")
    samp = "sample_submission.csv"          
    if os.path.exists(samp):
        write_submission(preds_by_well, samp, out="submission.csv")
