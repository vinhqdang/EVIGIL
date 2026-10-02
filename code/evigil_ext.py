"""Extensions of the curvature monitor (see evigil.py):
 - unknown common scale kappa of the published standard errors (scale-invariant e-process)
 - regime-specific slopes and guard waves around announced regime changes
 - mixture over the prior scale tau"""
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from evigil import TAU, THRESH

def apply_guard(reg, guard=1):
    """Give the waves c-guard..c+guard around each announced regime change c (c = first wave of the
    new regime) a regime of their own, so that each is absorbed by its own level. This protects
    against a regime label that is off by up to `guard` waves, at the cost of those waves."""
    reg = np.asarray(reg).copy(); out = reg.copy(); nxt = reg.max() + 1
    ch = [j for j in range(1, len(reg)) if reg[j] != reg[j - 1]]
    for j in ch:
        for k in range(max(0, j - guard), min(len(reg), j + guard + 1)):
            out[k] = nxt; nxt += 1
    return out

def design_cols(reg, t, regime_slopes=False):
    reg = np.asarray(reg); cols = []
    tt = t - t.mean()
    for g in np.unique(reg):
        m = (reg == g).astype(float)
        cols.append(m)
        if regime_slopes and m.sum() >= 1: cols.append(m * tt)
    if not regime_slopes: cols.append(tt)
    return np.column_stack(cols)

def projections(s, t, reg, regime_slopes=False, guard=0, pool=False, nmin=3):
    """For n = nmin..N return dict of arrays: C (row = map y -> A_n), B, P (n x n quadratic form
    giving RSS0 = y' P y) and nu = n - rank(D_n)."""
    N = len(s)
    reg = np.zeros(N, int) if pool else (apply_guard(reg, guard) if guard else np.asarray(reg))
    C = np.zeros((N, N)); B = np.zeros(N); nu = np.zeros(N, int); P = [None] * N
    for n in range(nmin, N + 1):
        w = 1 / s[:n]**2; sw = np.sqrt(w)
        D = design_cols(reg[:n], t[:n], regime_slopes)
        q = (t[:n] - t[:n].mean())**2 / 2
        X = D * sw[:, None]
        Q, Rr = np.linalg.qr(X); rank = np.linalg.matrix_rank(X)
        H = X @ np.linalg.pinv(X)                   # projector in the weighted space
        Mw = np.eye(n) - H
        qt = Mw @ (q * sw)                          # weighted residual of q
        C[n - 1, :n] = qt * sw
        B[n - 1] = qt @ qt
        P[n - 1] = sw[:, None] * Mw * sw[None, :]   # y' P y = RSS0 (Mw symmetric idempotent)
        nu[n - 1] = n - rank
    return dict(C=C, B=B, P=P, nu=nu)

# ---------- scale-invariant (unknown kappa) e-process, numerical mixture
_XR = np.linspace(-1.0, 1.0, 80)

def _logJ(a, nu):
    """log of int_0^inf r^(nu-1) exp(-r^2/2 + a r) dr, vectorized over a (shape (...,)), nu scalar array broadcastable."""
    a = np.asarray(a, float); nu = np.asarray(nu, float)
    rstar = 0.5 * (a + np.sqrt(a * a + 4.0 * np.maximum(nu - 1.0, 1e-9)))
    lo = np.maximum(rstar - 12.0, 1e-9); hi = rstar + 12.0
    r = lo[..., None] + (hi - lo)[..., None] * (_XR + 1.0)[None, :] / 2.0 if lo.ndim else lo + (hi - lo) * (_XR + 1) / 2
    f = (nu[..., None] - 1.0) * np.log(r) - r * r / 2.0 + a[..., None] * r
    return logsumexp(f, axis=-1) + np.log((hi - lo) / 2.0 * (_XR[1] - _XR[0]))

_PQ = np.linspace(0.0, 6.0, 60)      # prior nodes in units of tau (0 to 6 tau)
_K = np.arange(-7.0, 7.01, 0.5)

def log_e_unknown(A, B, rss0, nu, tau=TAU):
    """log E for the one-sided (gamma > 0) mixture with unknown scale: A, B, rss0, nu are arrays of equal shape."""
    A = np.asarray(A, float).ravel(); B = np.asarray(B, float).ravel()
    rss0 = np.asarray(rss0, float).ravel(); nu = np.asarray(nu, float).ravel()
    out = np.zeros(len(A))
    ok = (B > 0) & (nu >= 2) & (rss0 > 0)
    idx = np.where(ok)[0]
    for c in range(0, len(idx), 400):
        i = idx[c:c + 400]
        a_, b_, r_, n_ = A[i], B[i], rss0[i], nu[i]
        dhat = a_ * np.sqrt(n_) / (b_ * np.sqrt(r_))
        post = np.maximum(dhat[:, None] + _K[None, :] / np.sqrt(b_)[:, None], 0.0)
        prior = tau * _PQ[None, :] * np.ones((len(i), 1))
        d = np.sort(np.concatenate([post, prior, np.zeros((len(i), 1))], axis=1), axis=1)
        s_ = a_ / np.sqrt(r_)
        lj = _logJ(d * s_[:, None], n_[:, None]) - _logJ(np.zeros(len(i)), n_)[:, None]
        logLR = -d * d * b_[:, None] / 2.0 + lj
        logpi = norm.logpdf(d, scale=tau) + np.log(2.0)
        f = logLR + logpi
        w = np.diff(d, axis=1)
        trap = 0.5 * (np.exp(f[:, 1:] - f.max(1, keepdims=True)) + np.exp(f[:, :-1] - f.max(1, keepdims=True))) * w
        out[i] = f.max(1) + np.log(np.maximum(trap.sum(1), 1e-300))
    return out

TAUS = (0.0025, 0.005, 0.01, 0.02, 0.04)
def log_e_unknown_mix(A, B, rss0, nu, taus=TAUS):
    L = np.stack([log_e_unknown(A, B, rss0, nu, t) for t in taus])
    return logsumexp(L, axis=0) - np.log(len(taus))

def log_e_known(A, B, tau=TAU, gamma0=0.0):
    a = B + tau**-2; Ap = A - gamma0 * B; z = Ap / np.sqrt(a)
    return np.log(2 / (tau * np.sqrt(a))) + norm.logcdf(z) + z * z / 2

def log_e_mix(A, B, taus=(0.0025, 0.005, 0.01, 0.02, 0.04), gamma0=0.0):
    """Equal-weight mixture over a grid of prior scales (an e-process)."""
    L = np.stack([log_e_known(A, B, t, gamma0) for t in taus])
    return logsumexp(L, axis=0) - np.log(len(taus))

def surveil_ext(y, s, t, reg, mode="known", tau=TAU, thresh=THRESH, regime_slopes=False, guard=0,
                pool=False, gamma0=0.0, min_nu=2):
    """Epoch sequence with restarts for the extended monitors. mode: known | unknown | mix.
    Returns (declarations, trajectories); each trajectory is a list of (wave, E+, E-)."""
    N = len(y); start = 0; decl, traj = [], []; info = {}
    regv = np.zeros(N, int) if pool else (apply_guard(reg, guard) if guard else np.asarray(reg))
    while start + 3 <= N:
        pr = projections(s[start:], t[start:], regv[start:], regime_slopes=regime_slopes, pool=False)
        p = []
        for n in range(3, N - start + 1):
            B = pr["B"][n - 1]
            if B <= 1e-12 or (mode == "unknown" and pr["nu"][n - 1] < min_nu):
                p.append((start + n - 1, 1.0, 1.0)); continue
            yy = y[start:start + n]; A = pr["C"][n - 1][:n] @ yy
            if mode == "known":
                lp, lm = log_e_known(A, B, tau, gamma0), log_e_known(-A, B, tau, gamma0)
            elif mode == "mix":
                lp, lm = log_e_mix(A, B, gamma0=gamma0), log_e_mix(-A, B, gamma0=gamma0)
            else:
                rss = yy @ pr["P"][n - 1][:n, :n] @ yy; nu = pr["nu"][n - 1]
                f = log_e_unknown_mix if mode == "unknown_mix" else (lambda a, b, r, v: log_e_unknown(a, b, r, v, tau))
                lp = f([A], [B], [rss], [nu])[0]; lm = f([-A], [B], [rss], [nu])[0]
                info[start + n - 1] = (A, B, rss, nu)
            p.append((start + n - 1, float(np.exp(min(lp, 700))), float(np.exp(min(lm, 700)))))
        traj.append((start, p))
        hit = next((r for r in p if max(r[1], r[2]) >= thresh), None)
        if hit is None: break
        rec = dict(start=start, wave=hit[0], dir="accel" if hit[1] >= thresh else "decel", E=max(hit[1], hit[2]))
        if hit[0] in info: rec.update(A=info[hit[0]][0], B=info[hit[0]][1], rss=info[hit[0]][2], nu=info[hit[0]][3])
        decl.append(rec); start = hit[0] + 1
    return decl, traj
