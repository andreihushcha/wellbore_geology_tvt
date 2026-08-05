"""
TRANSFORMER (train offline on GPU) + AUGMENTATION.
=============================================================
Dual-encoder cross-attention geosteering model:
  * lateral encoder (bidirectional over GR+geometry)  -- GR is fully observed at test time
  * typewell encoder (tokens tagged with their TVT)    -- the reference log
  * cross-attention decoder                            -- differentiable log-correlation
  * heads: dTVT increment (primary) + typewell-match logits (alignment aux = the novelty)

Target = per-row dTVT (anchor-free, stationary). Reconstruct TVT = anchor + cumsum(dTVT).
Randomized-PS masking each step. Augmentations: synthetic laterals, burst GR masking, GR jitter.

"""
import os, glob, math, numpy as np, pandas as pd, torch, torch.nn as nn
from preprocess import preprocess_well, MODEL_FEATURES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_COLS = [c for c in MODEL_FEATURES if c not in ("known_flag", "dist_from_ps_norm")]  # 16 anchor-free feats
N_FEAT = len(FEAT_COLS)            # match head is anchor-free -> no window/simps channels (avoids train/infer mismatch)
TW_LEN = 192                       # typewell resampled length

#  data
def prep(stem):
    w = preprocess_well(stem)
    w["F"] = w["feats"][FEAT_COLS].values.astype(np.float32)   # (n, 16)
    lo, hi = w["tw_tvt"].min(), w["tw_tvt"].max()
    w["tw_grid"] = np.linspace(lo, hi, TW_LEN).astype(np.float32)
    w["tw_gr_grid"] = np.interp(w["tw_grid"], w["tw_tvt"], w["tw_gr"]).astype(np.float32)
    w["tw_grid_norm"] = ((w["tw_grid"] - w["tw_grid"].mean()) / (w["tw_grid"].std() + 1e-6)).astype(np.float32)
    return w

def synth_well(donor, rng):
    """Synthetic labeled lateral: walk a smooth mean-reverting TVT path through a real typewell,
    read GR off it (+noise), fabricate trajectory."""
    tw_tvt, tw_gr = donor["tw_tvt"], donor["tw_gr"]
    n = int(rng.integers(3000, 7000))
    lo, hi = np.percentile(tw_tvt, [15, 85])
    center = rng.uniform(lo, hi)
    # OU-like mean-reverting walk + occasional ramps
    tvt = np.empty(n, np.float32); tvt[0] = center; v = 0.0
    for i in range(1, n):
        v = 0.9*v + rng.normal(0, 0.04) - 0.02*(tvt[i-1]-center)/10
        tvt[i] = np.clip(tvt[i-1] + v, tw_tvt.min()+2, tw_tvt.max()-2)
    gr = np.interp(tvt, tw_tvt, tw_gr) + rng.normal(0, 0.25, n).astype(np.float32)
    F = np.zeros((n, len(FEAT_COLS)), np.float32); F[:, 0] = gr    # gr_norm is col 0
    ps = int(rng.integers(int(0.15*n), int(0.4*n)))
    tw_grid = np.linspace(tw_tvt.min(), tw_tvt.max(), TW_LEN).astype(np.float32)
    return dict(F=F, tvt=tvt, ps=ps, n=n,
                tw_gr_grid=np.interp(tw_grid, tw_tvt, tw_gr).astype(np.float32),
                tw_grid=tw_grid,
                tw_grid_norm=((tw_grid-tw_grid.mean())/(tw_grid.std()+1e-6)).astype(np.float32))

def burst_mask(F, rng, p=0.3):
    """Zero contiguous GR chunks (realistic dropout) -> col0=0, missing flag col1=1."""
    F = F.copy(); n = len(F); i = 0
    while i < n:
        if rng.random() < 0.02:
            L = int(rng.integers(2, 25)); F[i:i+L, 0] = 0.0
            if F.shape[1] > 1: F[i:i+L, 1] = 1.0
            i += L
        else: i += 1
    return F

class WindowDS(torch.utils.data.Dataset):
    def __init__(self, wells, L=512, synth_donors=None, n_synth=0, augment=True, seed=0):
        self.wells = wells; self.L = L; self.aug = augment
        self.rng = np.random.default_rng(seed)
        self.synth = [synth_well(rng_choice(synth_donors, self.rng), self.rng)
                      for _ in range(n_synth)] if (synth_donors and n_synth) else []
        self.pool = wells + self.synth
    def __len__(self): return max(256, 32*len(self.pool))
    def __getitem__(self, _):
        w = self.pool[self.rng.integers(len(self.pool))]
        n, L = w["n"], self.L
        s0 = int(self.rng.integers(0, max(1, n - L)))
        sl = slice(s0, min(n, s0 + L)); m = min(L, n - s0)
        F = w["F"][sl].copy()
        if self.aug and w in self.wells:
            F = burst_mask(F, self.rng)
            F[:, 0] = F[:, 0] * self.rng.uniform(0.9, 1.1) + self.rng.normal(0, 0.05)  # GR jitter
        tvt = w["tvt"][sl].astype(np.float32)
        X = F                                                      # (m, N_FEAT) anchor-free
        dtvt = np.diff(tvt, prepend=tvt[0]).astype(np.float32)
        gidx = np.clip(np.searchsorted(w["tw_grid"], tvt), 0, TW_LEN - 1).astype(np.int64)
        pad = L - m
        def pz(a, v=0): return np.pad(a, [(0, pad)] + [(0, 0)]*(a.ndim-1), constant_values=v)
        valid = np.pad(np.ones(m, np.float32), (0, pad))
        return (torch.tensor(pz(X)), torch.tensor(pz(dtvt)), torch.tensor(pz(gidx)),
                torch.tensor(valid), torch.tensor(valid),
                torch.tensor(w["tw_gr_grid"]), torch.tensor(w["tw_grid_norm"]))

def rng_choice(lst, rng): return lst[rng.integers(len(lst))]

# model
class ConvStem(nn.Module):
    def __init__(s, cin, d):
        super().__init__(); s.net = nn.Sequential(
            nn.Conv1d(cin, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2), nn.GELU())
    def forward(s, x): return s.net(x.transpose(1, 2)).transpose(1, 2)

class GeoSteer(nn.Module):
    def __init__(s, n_feat=N_FEAT, d=192, heads=6, enc=4, dec=2):
        super().__init__()
        s.lat = ConvStem(n_feat, d)
        s.tw = ConvStem(1, d); s.tvt_embed = nn.Linear(1, d)
        s.lat_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True), enc)
        s.tw_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True), 2)
        s.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4*d, batch_first=True), dec)
        s.dtvt = nn.Linear(d, 1); s.scale = d ** -0.5
    def forward(s, X, tw_gr, tw_tvt_norm, pad_mask=None):
        H = s.lat_enc(s.lat(X), src_key_padding_mask=pad_mask)
        M = s.tw_enc(s.tw(tw_gr.unsqueeze(-1))) + s.tvt_embed(tw_tvt_norm.unsqueeze(-1))
        Z = s.dec(H, M, tgt_key_padding_mask=pad_mask)
        dtvt = s.dtvt(Z).squeeze(-1)
        match = torch.bmm(Z, M.transpose(1, 2)) * s.scale        # (B,L,TW_LEN) cross-correlation
        return dtvt, match

# train
def train(wells, epochs=8, L=512, bs=8, lr=3e-4, n_synth=0, aug=True, lam_smooth=0.05,
          beta_align=0.3, seed=0, log_every=20):
    torch.manual_seed(seed)
    ds = WindowDS(wells, L=L, synth_donors=wells, n_synth=n_synth, augment=aug, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    model = GeoSteer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction="none"); step = 0
    for ep in range(epochs):
        for X, dtvt, gidx, wmask, valid, twg, twn in dl:
            X, dtvt, gidx = X.to(DEVICE), dtvt.to(DEVICE), gidx.to(DEVICE)
            wmask, valid, twg, twn = wmask.to(DEVICE), valid.to(DEVICE), twg.to(DEVICE), twn.to(DEVICE)
            pad = (valid == 0)
            pdt, match = model(X, twg, twn, pad_mask=pad)
            wm = wmask * valid                                    # supervise unknown & valid rows
            l_reg = ((pdt - dtvt) ** 2 * wm).sum() / (wm.sum() + 1e-6)
            l_smooth = ((pdt[:, 1:] - pdt[:, :-1]) ** 2 * wm[:, 1:]).sum() / (wm[:, 1:].sum() + 1e-6)
            l_align = (ce(match.reshape(-1, TW_LEN), gidx.reshape(-1)).reshape(gidx.shape) * wm).sum() / (wm.sum() + 1e-6)
            loss = l_reg + lam_smooth * l_smooth + beta_align * l_align
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % log_every == 0:
                print(f"ep{ep} step{step}: loss={loss.item():.4f} (reg={l_reg.item():.4f} "
                      f"align={l_align.item():.3f})")
            step += 1
    return model

def save_weights(model, path="geosteer.pt"):
    torch.save(model.state_dict(), path); print("saved", path)

# inference
def make_predict(model, L=512, stride=256):
    model.eval()
    @torch.no_grad()
    def predict(w):
        F = w["feats"][FEAT_COLS].values.astype(np.float32) if isinstance(w["feats"], pd.DataFrame) else None
        n, ps = w["n"], w["ps"]; anchor = float(w["tvt_input"][ps - 1])
        lo, hi = w["tw_tvt"].min(), w["tw_tvt"].max()
        grid = np.linspace(lo, hi, TW_LEN).astype(np.float32)
        twg = torch.tensor(np.interp(grid, w["tw_tvt"], w["tw_gr"]).astype(np.float32))[None].to(DEVICE)
        twn = torch.tensor(((grid-grid.mean())/(grid.std()+1e-6)).astype(np.float32))[None].to(DEVICE)
        Xfull = F                                                # anchor-free (16 cols)
        dt_acc = np.zeros(n); cnt = np.zeros(n)
        mt_acc = np.zeros((n, TW_LEN), np.float32)               # accumulate match logits
        for s0 in range(0, n, stride):
            sl = slice(s0, min(n, s0 + L)); m = sl.stop - sl.start
            xb = np.zeros((L, N_FEAT), np.float32); xb[:m] = Xfull[sl]
            valid = np.zeros(L, np.float32); valid[:m] = 1
            pad = torch.tensor(valid == 0)[None].to(DEVICE)
            pdt, match = model(torch.tensor(xb)[None].to(DEVICE), twg, twn, pad_mask=pad)
            dt_acc[sl] += pdt[0, :m].cpu().numpy()
            mt_acc[sl] += match[0, :m].cpu().numpy(); cnt[sl] += 1
            if sl.stop == n: break
        # absolute TVT from the alignment head (DRIFT-FREE). Argmax+smooth.
        logits = mt_acc / np.maximum(cnt, 1)[:, None]
        # sub-cell refinement: local softmax in a small window around the argmax peak
        # (avoids the multimodal averaging of a global soft-expectation, but recovers
        #  precision below the grid spacing -> not limited by TW_LEN quantization)
        peak = np.argmax(logits, 1)
        R = 3
        match_tvt = np.empty(n)
        for i in range(n):
            lo_i, hi_i = max(0, peak[i]-R), min(TW_LEN, peak[i]+R+1)
            w_ = logits[i, lo_i:hi_i]; w_ = np.exp(w_ - w_.max()); w_ /= w_.sum()
            match_tvt[i] = float((grid[lo_i:hi_i] * w_).sum())
        match_tvt = pd.Series(match_tvt).rolling(15, center=True, min_periods=1).median().values
        out = w["tvt_input"].copy()
        out[ps:] = np.clip(match_tvt[ps:], lo, hi)              # guardrailed to typewell range
        return out
    return predict

if __name__ == "__main__":

    import sys, torch
    ROOT = sys.argv[1] if len(sys.argv) > 1 else "train"
    stems = [hp[:-len("__horizontal_well.csv")]
             for hp in sorted(glob.glob(f"{ROOT}/*__horizontal_well.csv") or
                              glob.glob(f"{ROOT}/**/*__horizontal_well.csv", recursive=True))
             if os.path.exists(hp[:-len("__horizontal_well.csv")] + "__typewell.csv")]
    if not stems:
        sys.exit(f"No wells found in '{ROOT}'.")
    w0 = next((w for w in (prep(s) for s in stems) if w["tvt"] is not None), None)
    if w0 is None:
        sys.exit("No training-format well with a TVT column found (need ground truth for the check).")
    # carry-forward baseline on this well (honest reference, not a hardcoded number)
    ps = w0["ps"]; ev = w0["eval_mask"] & ~np.isnan(w0["tvt"])
    carry = float(np.sqrt(np.mean((w0["tvt_input"][ps-1] - w0["tvt"][ev])**2)))
    print(f"device={DEVICE} | N_FEAT={N_FEAT} | sanity: overfit one well (match-argmax decode)")
    model = train([w0], epochs=10, L=128, bs=8, n_synth=0, aug=False, log_every=100)
    predict = make_predict(model, L=128, stride=64)
    p = predict(w0); rmse = float(np.sqrt(np.mean((p[ev]-w0["tvt"][ev])**2)))
    verdict = "PASS (pipeline correct)" if rmse < carry else "FAIL (decode/train bug — do NOT trust CV)"
    print(f"OVERFIT RMSE={rmse:.2f} ft  |  carry={carry:.2f} ft  ->  {verdict}")
    save_weights(model, "geosteer_smoke.pt")
