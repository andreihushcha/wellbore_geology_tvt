"""
TRAINING DRIVER  (offline GPU training; NOT the Kaggle submission)
"""
import os, glob, argparse, numpy as np, torch
import transformer_train as T
from geosteering_cv import group_kfold
from features_and_models import CarryP, ViterbiP
from ensemble import guardrail

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# full-run defaults; bs sized for 8 GB VRAM (RTX 5070 laptop). 
FULL  = dict(epochs=40, L=512, bs=6,  lr=3e-4, n_synth_frac=0.5, k=5, log_every=200)
QUICK = dict(epochs=20, L=512, bs=6,  lr=3e-4, n_synth_frac=0.5, k=3, log_every=200)
SMOKE = dict(epochs=6,  L=128, bs=8,  lr=3e-4, n_synth_frac=0.0, k=2, log_every=100)

def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _train_with_oom_retry(wells, cfg, ns):
    """Train one model; if CUDA OOMs, retry once at half batch (8 GB safety net)."""
    try:
        return T.train(wells, epochs=cfg["epochs"], L=cfg["L"], bs=cfg["bs"], lr=cfg["lr"],
                       n_synth=ns, aug=True, log_every=cfg["log_every"])
    except torch.cuda.OutOfMemoryError:
        _empty_cache()
        half = max(1, cfg["bs"] // 2)
        print(f"  [OOM] retrying at bs={half} (was {cfg['bs']}) ...")
        return T.train(wells, epochs=cfg["epochs"], L=cfg["L"], bs=half, lr=cfg["lr"],
                       n_synth=ns, aug=True, log_every=cfg["log_every"])

def load_all(data_dir):
    stems = [hp[:-len("__horizontal_well.csv")]
             for hp in sorted(glob.glob(f"{data_dir}/**/*__horizontal_well.csv", recursive=True))
             if os.path.exists(hp[:-len("__horizontal_well.csv")] + "__typewell.csv")]
    print(f"discovered {len(stems)} wells in {data_dir}")
    return [T.prep(s) for s in stems]

def _pooled_rmse(pred_fn, wells):
    err = []
    for w in wells:
        p = pred_fn(w); m = w["eval_mask"] & ~np.isnan(w["tvt"]) & ~np.isnan(p)
        err.append(p[m] - w["tvt"][m])
    e = np.concatenate(err) if err else np.array([np.nan])
    return float(np.sqrt(np.nanmean(e ** 2)))

def cv_transformer(wells, cfg):
    """GroupKFold: train transformer on train folds, score held-out wells. Compare to baselines.
    This is the number that decides whether the transformer ships (vs the Phase-B best)."""
    ids = [w["stem"].split("/")[-1] for w in wells]
    by = {w["stem"].split("/")[-1]: w for w in wells}
    folds = group_kfold(ids, k=min(cfg["k"], len(ids)), seed=0)
    tf_err, ca_err, vi_err = [], [], []
    # Viterbi baseline must use the SWEPT penalties (Phase-B best = 0.4/0.6 -> 14.717 = carry),
    vit = ViterbiP(prior_pen=0.4, step_pen=0.6)
    for fi, (tr, va) in enumerate(folds):
        trw = [by[i] for i in tr]; vaw = [by[i] for i in va]
        if not trw or not vaw: continue
        ns = int(cfg["n_synth_frac"] * len(trw))
        print(f"\n--- fold {fi+1}/{len(folds)}: train {len(trw)} / val {len(vaw)} (n_synth={ns}) ---")
        model = _train_with_oom_retry(trw, cfg, ns)
        pred = T.make_predict(model, L=cfg["L"], stride=cfg["L"]//2)
        tfp = lambda w: guardrail(w, pred(w))               # guardrailed transformer
        for w in vaw:
            for store, fn in [(tf_err, tfp),
                              (ca_err, CarryP().predict),
                              (vi_err, vit.predict)]:
                p = fn(w); m = w["eval_mask"] & ~np.isnan(w["tvt"]) & ~np.isnan(p)
                store.append(p[m]-w["tvt"][m])
        del model, pred; _empty_cache()                     # free VRAM before next fold (8 GB)
    def rmse(es): e = np.concatenate(es) if es else np.array([np.nan]); return np.sqrt(np.nanmean(e**2))
    r_ca, r_vi, r_tf = rmse(ca_err), rmse(vi_err), rmse(tf_err)
    bar = min(r_ca, r_vi)                                   # the honest Phase-B best to beat
    print("\n==================  CV pooled RMSE (held-out)  ==================")
    print(f"  carry-forward        : {r_ca:.3f}")
    print(f"  viterbi corr (0.4/0.6): {r_vi:.3f}")
    print(f"  TRANSFORMER          : {r_tf:.3f}   <- ships only if it beats {bar:.3f}")
    print(f"  VERDICT: transformer {'BEATS' if r_tf < bar else 'does NOT beat'} the Phase-B best "
          f"({r_tf:.3f} vs {bar:.3f})")
    print("=================================================================")

def train_final(wells, cfg, out):
    ns = int(cfg["n_synth_frac"] * len(wells))
    print(f"\ntraining FINAL model on all {len(wells)} wells (n_synth={ns}) ...")
    _empty_cache()
    model = _train_with_oom_retry(wells, cfg, ns)
    torch.save(model.state_dict(), out)
    print(f"saved weights -> {out}\n"
          f"NEXT: upload {out} as a Kaggle Dataset, then in the Kaggle inference notebook:\n"
          f"  from transformer_train import GeoSteer, make_predict\n"
          f"  m = GeoSteer(); m.load_state_dict(torch.load('/kaggle/input/<your-dataset>/geosteer.pt'))\n"
          f"  PREDICTOR = make_predict(m)   # in kaggle_submission.py")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATA_DIR", "train"))
    ap.add_argument("--out",  default=os.environ.get("OUT", "geosteer.pt"))
    ap.add_argument("--smoke", action="store_true", help="tiny wiring check")
    ap.add_argument("--quick", action="store_true", help="20 epochs / k=3: fast honest verdict")
    ap.add_argument("--bs", type=int, default=None, help="override batch size (lower if OOM)")
    a = ap.parse_args()
    cfg = SMOKE if a.smoke else (QUICK if a.quick else FULL)
    if a.bs: cfg = dict(cfg, bs=a.bs)

    tag = "SMOKE" if a.smoke else ("QUICK" if a.quick else "FULL")
    print(f"device={T.DEVICE} | cfg={tag} | bs={cfg['bs']} L={cfg['L']} epochs={cfg['epochs']} k={cfg['k']}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"gpu={torch.cuda.get_device_name(0)} | VRAM free {free/1e9:.1f}/{total/1e9:.1f} GB")
    else:
        print("WARNING: device is CPU -- training will be very slow. Check your torch+cuda install.")

    wells = load_all(a.data)
    if not wells:
        raise SystemExit(
            f"no wells found under {a.data!r}. Check the --data path. "
            f"On Git Bash, use a RELATIVE path like 'train' (a leading /content/... is rewritten).")

    cv_transformer(wells, cfg)                              # honest held-out comparison
    train_final(wells, cfg, a.out)                          # final model for submission
