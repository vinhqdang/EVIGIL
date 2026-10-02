"""Replication script for Section 5.6 (the long view, 1990-2026) and Table 4.
Inputs: internet_users_us.csv (World Bank/ITU via FRED, ITNETUSERP2USA) and
ecommerce_share_us.csv (Census via FRED, ECOMPCTSA), included as supplementary
data exactly as retrieved. Plug-in-variance monitor per Sec. 3.5 of the paper.
Usage: python reproduce_retro.py"""
import numpy as np, csv, json
from statistics import NormalDist
ND = NormalDist(); TAU, THRESH, W = 1.0, 40.0, 8

def e1(S, n, tau=TAU):
    a = n + 1/tau**2
    logE = (np.log(2/(tau*np.sqrt(a)))
            + np.log(max(ND.cdf(S/np.sqrt(a)), 1e-300)) + S**2/(2*a))
    return np.exp(min(logE, 50))          # cap for reporting

def mad_scale(r, floor):
    r = np.array(r[-W:])
    if len(r) < 3:
        return floor
    return max(floor, 1.4826*np.median(np.abs(r - np.median(r))))

def wls_slope(y, idx):
    x = np.arange(len(idx), dtype=float); yy = y[idx]
    X = np.column_stack([np.ones(len(idx)), x])
    b, *_ = np.linalg.lstsq(X, yy, rcond=None)
    res = yy - X@b
    se = np.sqrt(np.sum(res**2)/max(len(idx)-2, 1)/np.sum((x-x.mean())**2))
    return float(b[1]), float(se), list(np.diff(yy) - b[1])

def surveil(labels, y, floor):
    T = len(y); out = []; t0 = 1; beta, bse = 0.0, 0.0; epoch = 1; resids = []
    while t0 < T:
        Sup = Sdn = 0.0; n = 0; det = None
        for t in range(t0, T):
            d = y[t] - y[t-1]
            if len(resids) < 3:                 # epoch-1 warm-up only
                resids.append(d - beta); continue
            sig = np.sqrt(mad_scale(resids, floor)**2 + bse**2)
            z = (d - beta)/sig
            n += 1; Sup += z; Sdn -= z
            resids.append(d - beta)
            Eu, Ed = e1(Sup, n), e1(Sdn, n)
            if Eu >= THRESH or Ed >= THRESH:
                det = dict(epoch=epoch, label=labels[t],
                           dir='up' if Eu >= THRESH else 'down',
                           e=float(max(Eu, Ed)), baseline=round(beta, 3), t=t)
                break
        if det is None:
            out.append(dict(epoch=epoch, label='(end of sample)', dir='none',
                            e=float(max(e1(Sup, max(n, 1)), e1(Sdn, max(n, 1)))),
                            baseline=round(beta, 3)))
            break
        out.append(det); a0 = det['t'] + 1
        if a0 + W >= T:
            break
        beta, bse, resids = wls_slope(y, list(range(a0, a0 + W)))
        t0 = a0 + W; epoch += 1
    return out

for name, path, col, floor in [
        ("Internet users (annual, 1990-2024)", "internet_users_us.csv", 1, 0.8),
        ("E-commerce share (quarterly, 1999Q4-2026Q1)", "ecommerce_share_us.csv", 1, 0.08)]:
    rows = list(csv.reader(open(path)))[1:]
    y = np.array([float(r[col]) for r in rows])
    labels = [r[0][:7] for r in rows]
    print(f"=== {name} ===")
    for s in surveil(labels, y, floor):
        e = f">1e4" if s['e'] > 1e4 else f"{s['e']:.1f}"
        print(f"  epoch {s['epoch']}: baseline {s['baseline']:+.3f} -> "
              f"{s['dir']:>4} @ {s['label']}  (E={e})")
