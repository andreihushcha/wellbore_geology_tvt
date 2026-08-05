"""
ENSEMBLE + PHYSICAL GUARDRAILS.
"""
import numpy as np

def guardrail(w, tvt, max_dtvt=0.6):
    """Clip to typewell range, cap per-ft slope, anchor at PS."""
    lo, hi = w["tw_tvt"].min(), w["tw_tvt"].max()
    ps = w["ps"]; out = tvt.copy()
    out[ps:] = np.clip(out[ps:], lo, hi)
    anchor = float(w["tvt_input"][ps-1]); prev = anchor          # slope cap + anchor
    for i in range(ps, len(out)):
        step = np.clip(out[i]-prev, -max_dtvt, max_dtvt); prev = prev+step; out[i] = prev
    return out

def blend(w, members, weights=None, conf=None):
    """members: list of full-length arrays. weights: per-member scalar or per-row array.
    """
    M = np.stack([m for m in members], 0)                        # (K, n)
    if conf is not None:
        Wt = np.stack(conf, 0); Wt = Wt / (Wt.sum(0, keepdims=True)+1e-9)
    else:
        wv = np.ones(len(members)) if weights is None else np.asarray(weights, float)
        wv = wv/wv.sum(); Wt = wv[:, None]
    out = (M*Wt).sum(0)
    out[:w["ps"]] = w["tvt_input"][:w["ps"]]                     # keep known region exact
    return guardrail(w, out)



if __name__ == "__main__":

    w = dict(ps=3, tw_tvt=np.array([0.,100.]), tvt_input=np.array([10.,10.,10.,np.nan,np.nan,np.nan]))
    a = np.array([10,10,10, 12, 40, 5], float)   # a wild member (jumps)
    b = np.array([10,10,10, 11, 12, 13], float)
    out = blend(w, [a, b], weights=[0.5,0.5])
    print("blended+guardrailed:", np.round(out,2), "| known region preserved:", np.allclose(out[:3],10))
