"""Core routines: invariant mixture e-process for curvature (change in growth rate)
against a locally linear null with unknown level and slope."""
import numpy as np, pandas as pd
from scipy.stats import norm

TAU = 0.01          # prior scale on curvature, pp per wave^2
THRESH = 40.0       # 2/alpha with alpha = 0.05 (two one-sided monitors)

def load_national(path="data/btos_ai_national.csv"):
    d = pd.read_csv(path, parse_dates=["collection_end"]).sort_values("collection_end")
    d = d.reset_index(drop=True)
    d["t"] = (d.collection_end - d.collection_end[0]).dt.days / 14.0
    d["reg"] = (d.wording == "revised").astype(int)
    return d

def _resid(v, D, w):
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(D * sw[:, None], v * sw, rcond=None)
    return v - D @ b

def score(y, s, t, reg):
    """(A, B): score and information of the curvature coefficient after projecting
    out regime-specific intercepts and a common linear trend (WLS, weights 1/s^2)."""
    w = 1.0 / s**2
    tt = t - t.mean()
    cols = [(reg == g).astype(float) for g in np.unique(reg)] + [tt]
    D = np.column_stack(cols)
    x = tt**2 / 2.0
    xt, yt = _resid(x, D, w), _resid(y, D, w)
    return float((w * xt * yt).sum()), float((w * xt * xt).sum())

def evalue(A, B, tau=TAU):
    """Half-normal mixture over the curvature: closed form (log scale for stability)."""
    a = B + tau**-2
    z = A / np.sqrt(a)
    return float(np.exp(np.log(2.0 / (tau * np.sqrt(a))) + norm.logcdf(z) + z * z / 2.0))

def path(y, s, t, reg, start, tau=TAU, pool_regimes=False, min_n=3):
    """E^+ (acceleration) and E^- (deceleration) for waves start..end of one epoch."""
    out = []
    for n in range(start + min_n, len(y) + 1):
        sl = slice(start, n)
        r = np.zeros(n - start, int) if pool_regimes else reg[sl]
        A, B = score(y[sl], s[sl], t[sl], r)
        out.append((n - 1, evalue(A, B, tau) if B > 0 else 1.0,
                    evalue(-A, B, tau) if B > 0 else 1.0))
    return out

def surveil(y, s, t, reg, tau=TAU, thresh=THRESH, pool_regimes=False, stop=None):
    """Sequence of epochs; after each declaration the monitor restarts at the next wave."""
    stop = len(y) if stop is None else stop
    start, decl, traj = 0, [], []
    while start + 3 <= stop:
        p = path(y[:stop], s[:stop], t[:stop], reg[:stop], start, tau, pool_regimes)
        traj.append((start, p))
        hit = next((r for r in p if max(r[1], r[2]) >= thresh), None)
        if hit is None:
            break
        decl.append(dict(start=start, wave=hit[0], dir="accel" if hit[1] >= thresh else "decel",
                         E=max(hit[1], hit[2])))
        start = hit[0] + 1
    return decl, traj
