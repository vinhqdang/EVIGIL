"""Universal e-process for changes in a published survey series when the level is rough.

Null H0 (class of Gaussian laws indexed by theta = (kappa^2, r)):
    y = X c + L + e,   e ~ N(0, kappa^2 diag(s^2)),   L_j = L_1 + sum_{i<=j} eta_i,  eta_i ~ N(0, kappa^2 r dt_i),
with X = [regime levels, time]; c (regime levels and slope), kappa^2 > 0 and r in the grid R0 unknown.
So y ~ N(X c, kappa^2 (D + r W)),  D = diag(s^2),  W_jk = min(T_j, T_k),  T_j = t_j - t_1.

Alternatives: after wave k the slope changes by delta (family "slope"), or the level jumps by delta
(family "level"), for an unknown k.  The e-process is

    E_n = Q_n(R_n) / sup_{theta in H0} P_theta(R_n),

R_n = y_{1:n} modulo span(X_n) (REML density), Q_n a fixed Gaussian mixture over (r, kappa^2), family,
k and delta.  E_n <= Q_n / P_theta*, a nonnegative martingale under the true theta*; Ville gives validity.
"""
import numpy as np
from scipy.special import logsumexp

def rw_cov(t):
    T = t - t[0]
    return np.minimum.outer(T, T)

class UI:
    def __init__(self, s, t, reg=None, nmin=4, kmin=3,
                 r_rel=(0.0, 0.5, 2.0, 8.0, 32.0), kap2=(0.5, 1.0, 2.0, 4.0),
                 tau_slope=(0.1, 0.2, 0.4, 0.8), tau_level=(0.5, 1.0, 2.0, 4.0),
                 families=("slope", "level"), r_null=None, extra=None, extra_in_prior=False):
        self.s, self.t = np.asarray(s, float), np.asarray(t, float)
        self.N = len(self.s); self.reg = np.zeros(self.N, int) if reg is None else np.asarray(reg)
        self.nmin, self.kmin = nmin, kmin
        med = float(np.median(self.s) ** 2)
        self.r_prior = med * np.array(r_rel); self.kap2 = np.array(kap2)
        self.tau = {"slope": np.array(tau_slope), "level": np.array(tau_level)}
        self.families = families
        self.r_null = (np.concatenate([[0.0], med * np.logspace(-2, 2.5, 46)]) if r_null is None else np.asarray(r_null))
        self.r_null = np.union1d(self.r_null, self.r_prior)
        self.rs = self.r_null.copy()
        self.extra = {} if extra is None else dict(extra)       # label -> function(t_n) giving an additional roughness covariance
        self.extra_in_prior = extra_in_prior
        self.ri = {float(r): i for i, r in enumerate(self.rs)}
        self.pre = {n: self._pre(n) for n in range(nmin, self.N + 1)}

    def _X(self, n):
        cols = [(self.reg[:n] == g).astype(float) for g in np.unique(self.reg[:n])]
        cols.append(self.t[:n] - self.t[:n].mean())
        return np.column_stack(cols)

    def _pre(self, n):
        X = self._X(n); p = X.shape[1]; nu = n - p
        D = np.diag(self.s[:n]**2); W = rw_cov(self.t[:n])
        Ws = [r * W for r in self.rs] + [f(self.t[:n]) for f in self.extra.values()]
        P1 = np.zeros((len(Ws), n, n)); ld = np.zeros(len(Ws)); lds = np.zeros(len(Ws))
        for i, Wi in enumerate(Ws):
            S1 = D + Wi; Si = np.linalg.inv(S1); XtSiX = X.T @ Si @ X
            P1[i] = Si - Si @ X @ np.linalg.solve(XtSiX, X.T @ Si)
            lds[i] = np.linalg.slogdet(S1)[1]; ld[i] = lds[i] + np.linalg.slogdet(XtSiX)[1]
        ks = np.arange(self.kmin, self.N + 1)
        Z = {"slope": np.stack([np.maximum(self.t[:n] - self.t[k - 1], 0.0) for k in ks]),
             "level": np.stack([(np.arange(n) >= k - 1).astype(float) for k in ks])}
        return dict(n=n, nu=nu, P1=P1, ld=ld, lds=lds, Z=Z, ks=ks)

    # ---- helpers on a data matrix Y (R, N)
    def _S(self, Y, n):
        y = Y[:, :n]; return np.einsum("rj,ijk,rk->ri", y, self.pre[n]["P1"], y)

    def null_idx(self):
        return [self.ri[float(r)] for r in self.r_null] + list(range(len(self.rs), len(self.rs) + len(self.extra)))

    def log_den(self, S, n, fixed=None):
        """log sup over kappa^2 and the null settings of the REML density (up to the common constant).
        fixed = (index, kappa^2) replaces the supremum by the density at a known setting (oracle benchmark)."""
        pr = self.pre[n]; nu = pr["nu"]
        ll = -0.5 * (nu * np.log(np.maximum(S, 1e-300) / nu) + nu + pr["ld"][None, :])
        if fixed is not None:
            i, kg = fixed; return -0.5 * (nu * np.log(kg) + pr["ld"][i] + S[:, i] / kg), ll
        return ll[:, self.null_idx()].max(1), ll

    def log_q(self, Y, n, S=None, parts=False):
        pr = self.pre[n]; nu = pr["nu"]; y = Y[:, :n]
        S = self._S(Y, n) if S is None else S
        comps, labels = [], []
        prior_idx = [self.ri[float(rg)] for rg in self.r_prior]
        if self.extra_in_prior: prior_idx += list(range(len(self.rs), len(self.rs) + len(self.extra)))
        for i in prior_idx:
            for kg in self.kap2:
                logP = -0.5 * (nu * np.log(kg) + pr["ld"][i] + S[:, i] / kg)
                for f in self.families:
                    Zp = pr["Z"][f] @ pr["P1"][i]
                    a = (y @ Zp.T) / kg; b = np.einsum("kj,kj->k", Zp, pr["Z"][f]) / kg
                    lbs = []
                    for tu in self.tau[f]:
                        t2 = tu**2
                        lbs.append(-0.5 * np.log1p(t2 * b)[None, :] + 0.5 * t2 * a**2 / (1 + t2 * b)[None, :])
                    lb = logsumexp(np.concatenate(lbs, 1), axis=1) - np.log(len(lbs) * a.shape[1])
                    comps.append(logP + lb); labels.append(f)
        C = np.stack(comps, 1)
        lq = logsumexp(C, axis=1) - np.log(C.shape[1])
        if parts:
            lab = np.array(labels); return lq, {f: logsumexp(C[:, lab == f], axis=1) - np.log((lab == f).sum()) for f in self.families}
        return lq

    def log_e(self, Y, parts=False, fixed=None):
        Y = np.atleast_2d(Y); out = np.full((Y.shape[0], self.N), -np.inf)
        pp = {f: np.full((Y.shape[0], self.N), -np.inf) for f in self.families}
        for n in range(self.nmin, self.N + 1):
            S = self._S(Y, n); den, _ = self.log_den(S, n, fixed)
            if parts:
                lq, lp = self.log_q(Y, n, S, parts=True)
                for f in self.families: pp[f][:, n - 1] = lp[f] - den
            else: lq = self.log_q(Y, n, S)
            out[:, n - 1] = lq - den
        return (out, pp) if parts else out

    def reml(self, y):
        """REML estimates (r, kappa^2) on the null grid and the profile log-likelihood over r (full sample)."""
        n = self.N; pr = self.pre[n]; S = self._S(np.atleast_2d(y), n)[0]
        _, ll = self.log_den(S[None, :], n)
        idx = [self.ri[float(r)] for r in self.r_null]
        llr = ll[0][idx]; j = int(llr.argmax()); i = idx[j]
        return dict(r=float(self.r_null[j]), kappa2=float(S[i] / pr["nu"]), profile=(self.r_null, llr))

    # ---- calibrated generalized likelihood-ratio comparator (maximum likelihood, not anytime-valid by itself)
    def log_glr(self, Y):
        """(R, N) array: 2 * max over family, change wave and roughness of the ML log-likelihood ratio, at each wave."""
        Y = np.atleast_2d(Y); out = np.full((Y.shape[0], self.N), -np.inf)
        for n in range(self.nmin, self.N + 1):
            pr = self.pre[n]; y = Y[:, :n]; idx = self.null_idx()
            S = np.einsum("rj,ijk,rk->ri", y, pr["P1"], y)[:, idx]
            l0 = (-0.5 * (n * np.log(np.maximum(S, 1e-300) / n) + pr["lds"][idx][None, :])).max(1)
            best = np.full(y.shape[0], -np.inf)
            for f in self.families:
                for jj, i in enumerate(idx):
                    Zp = pr["Z"][f] @ pr["P1"][i]; a = y @ Zp.T; b = np.einsum("kj,kj->k", Zp, pr["Z"][f])
                    Sa = np.maximum(S[:, jj][:, None] - a**2 / np.maximum(b, 1e-12)[None, :], 1e-300)
                    la = (-0.5 * (n * np.log(Sa / n) + pr["lds"][i])).max(1)
                    best = np.maximum(best, la)
            out[:, n - 1] = 2 * (best - l0)
        return out
