import numpy as np, pandas as pd
from scipy.stats import norm
nat=pd.read_csv("data/btos_ai_national.csv"); leg=nat[nat.wording=="legacy"]
s=leg.ai_use_se.to_numpy(); y=leg.ai_use_pct.to_numpy()
def E(S,n,tau=1.):
    a=n+1/tau**2; return 2/(tau*np.sqrt(a))*norm.cdf(S/np.sqrt(a))*np.exp(S**2/(2*a))
# empirical lag-1 autocorr of standardized increments
z=np.diff(y)/np.sqrt(s[1:]**2+s[:-1]**2); print("lag1 acf of z:",np.corrcoef(z[:-1],z[1:])[0,1])
rng=np.random.default_rng(1); R=20000; N=len(s); rej=0
for _ in range(R):
    pi=5+s*rng.standard_normal(N)       # flat latent, independent sampling noise
    zz=np.diff(pi)/np.sqrt(s[1:]**2+s[:-1]**2); S=np.cumsum(zz); n=np.arange(1,N)
    rej+= (E(S,n).max()>=40) or (E(-S,n).max()>=40)
print("type I (flat latent, indep noise):",rej/R)
# random-walk latent with innovations var = s_t^2+s_{t-1}^2 (the stated null)
rej=0
for _ in range(R):
    zz=rng.standard_normal(N-1); S=np.cumsum(zz); n=np.arange(1,N)
    rej+= (E(S,n).max()>=40) or (E(-S,n).max()>=40)
print("type I (stated iid z null):",rej/R)
