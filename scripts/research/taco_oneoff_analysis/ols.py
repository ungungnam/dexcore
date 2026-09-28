import numpy as np, pandas as pd
def ols(y, X, groups=None):
    """X: DataFrame with intercept column. Returns coef, cluster-robust SE, p (normal approx), r2."""
    Xm=X.values.astype(float); y=np.asarray(y,float)
    beta,*_=np.linalg.lstsq(Xm,y,rcond=None); e=y-Xm@beta
    XtX_inv=np.linalg.pinv(Xm.T@Xm)
    if groups is None:
        meat=(Xm*e[:,None]**2).T@Xm
    else:
        g=pd.factorize(groups)[0]; meat=np.zeros((Xm.shape[1],)*2)
        for k in np.unique(g):
            s=Xm[g==k].T@e[g==k]; meat+=np.outer(s,s)
        G=len(np.unique(g)); meat*=G/(G-1)
    V=XtX_inv@meat@XtX_inv; se=np.sqrt(np.diag(V))
    from math import erf
    p=[2*(1-0.5*(1+erf(abs(b/s)/np.sqrt(2)))) for b,s in zip(beta,se)]
    r2=1-e.var()/y.var()
    return pd.DataFrame(dict(coef=beta,se=se,p=p),index=X.columns), r2
def design(df, cols, cat=None):
    X=pd.DataFrame({'const':1.0},index=df.index)
    for c in cols: X[c]=df[c].astype(float)
    if cat:
        for c in cat:
            for v in sorted(df[c].unique())[1:]: X[f'{c}={v}']=(df[c]==v).astype(float)
    return X
