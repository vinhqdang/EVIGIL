"""Scale-exact universal e-process: invariance to the unknown scale kappa^2 as well as to the location nuisance.

The data are reduced to the maximal invariant u_n = R_n / |R_n| (direction of the residual vector modulo span(X_n)).
Under N(Xc, kappa^2 Sigma_r) the law of u_n is an angular central Gaussian that does not depend on (c, kappa^2):
    log p_r(u_n) = -1/2 ld_r - (nu/2) log S_r + (terms in u only),    S_r = y' P_r y.
A component of the alternative (family f, wave k, prior scale tau on delta ~ N(0, kappa^2 tau^2), roughness r_g) has
    log q(u_n) = log p_{r_g}(u_n) - 1/2 log(1 + tau^2 b) - (nu/2) log(1 - tau^2 a^2 / ((1 + tau^2 b) S_{r_g})),
a = z'P y, b = z'P z (P at kappa^2 = 1).  E_n = Q_n(u_n) / max_{r in R0} p_r(u_n); validity: Q_n / p_theta is a likelihood-ratio
martingale on the nested sigma-fields of u_n, and E_n <= Q_n / p_theta for every theta in the null class.
"""
import numpy as np
from scipy.special import logsumexp
from evigil_ui import UI

class UIX(UI):
    def __init__(self, *a, **k):
        k.setdefault("kap2", (1.0,))
        super().__init__(*a, **k)

    def log_den(self, S, n, fixed=None):
        pr = self.pre[n]; nu = pr["nu"]
        ll = -0.5 * (pr["ld"][None, :] + nu * np.log(np.maximum(S, 1e-300)))
        if fixed is not None:
            i = fixed; return ll[:, i], ll
        return ll[:, self.null_idx()].max(1), ll

    def log_q(self, Y, n, S=None, parts=False):
        pr = self.pre[n]; nu = pr["nu"]; y = Y[:, :n]
        S = self._S(Y, n) if S is None else S
        comps, labels = [], []
        for rg in self.r_prior:
            i = self.ri[float(rg)]; logP = -0.5 * (pr["ld"][i] + nu * np.log(np.maximum(S[:, i], 1e-300)))
            for f in self.families:
                Zp = pr["Z"][f] @ pr["P1"][i]
                a = y @ Zp.T; b = np.einsum("kj,kj->k", Zp, pr["Z"][f])
                lbs = []
                for tu in self.tau[f]:
                    t2 = tu**2; frac = np.clip(t2 * a**2 / ((1 + t2 * b)[None, :] * S[:, [i]]), 0, 1 - 1e-12)
                    lbs.append(-0.5 * np.log1p(t2 * b)[None, :] - 0.5 * nu * np.log1p(-frac))
                lb = logsumexp(np.concatenate(lbs, 1), axis=1) - np.log(len(lbs) * a.shape[1])
                comps.append(logP + lb); labels.append(f)
        C = np.stack(comps, 1); lq = logsumexp(C, axis=1) - np.log(C.shape[1])
        if parts:
            lab = np.array(labels); return lq, {f: logsumexp(C[:, lab == f], axis=1) - np.log((lab == f).sum()) for f in self.families}
        return lq

def kl_ceiling(s, t, reg, fam, k, delta, r_g, kap2_g, r_null):
    """inf over null (r, kappa^2) of KL( N(delta M z, kap2_g M Sigma_g M') || N(0, kappa^2 M Sigma_r M') ) on the residual space (nats)."""
    from evigil_ui import rw_cov
    n = len(s); u = UI(s, t, reg); X = u._X(n)
    Qm, _ = np.linalg.qr(np.column_stack([X, np.eye(n)]))   # orthonormal basis of col space of X first, then complement
    q = X.shape[1]; rank = np.linalg.matrix_rank(X); A = Qm[:, rank:rank + n - rank]
    z = np.maximum(t - t[k - 1], 0.0) if fam == "slope" else (np.arange(n) >= k - 1).astype(float)
    D = np.diag(s**2); W = rw_cov(t); m = delta * (A.T @ z); nu = A.shape[1]
    Sg = kap2_g * A.T @ (D + r_g * W) @ A; ldg = np.linalg.slogdet(Sg)[1]; best = np.inf
    for r in r_null:
        S1 = A.T @ (D + r * W) @ A; S1i = np.linalg.inv(S1)
        c = np.trace(S1i @ Sg) + m @ S1i @ m
        # infimum over kappa^2 of the KL: 1/2 [ nu log(c/nu) + logdet S1 - ldg ]
        best = min(best, 0.5 * (nu * np.log(c / nu) + np.linalg.slogdet(S1)[1] - ldg))
    return best

def power_ceiling(K, alpha=0.05):
    """Largest beta with kl(beta || alpha) <= K: no valid procedure with false-alarm prob <= alpha has power above it."""
    def kl(b):
        b = min(max(b, 1e-12), 1 - 1e-12); return b * np.log(b / alpha) + (1 - b) * np.log((1 - b) / (1 - alpha))
    lo, hi = alpha, 1 - 1e-12
    if kl(hi) <= K: return 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if kl(mid) <= K: lo = mid
        else: hi = mid
    return lo
