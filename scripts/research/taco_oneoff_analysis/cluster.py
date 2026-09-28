import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
df = pd.read_csv('/result/uhnam/dexcore/bimart_taco/contact_probe/gen/betas_per_sequence.csv')
L = df[[f'L{i}' for i in range(10)]].values; R = df[[f'R{i}' for i in range(10)]].values
for name, B in [('L',L),('R',R)]:
    key = [tuple(np.round(b,4)) for b in B]
    u = pd.Series(key).nunique()
    print(name, 'unique rounded-1e-4 vectors:', u)
    # pairwise distances distribution
    D = pdist(B); print(name, 'pdist quantiles', np.quantile(D,[0,0.001,0.01,0.05,0.1,0.5]))
    # nearest-neighbour distances
    S = squareform(D); np.fill_diagonal(S, np.inf)
    nn = S.min(1); print(name, 'NN dist quantiles', np.quantile(nn,[0,0.5,0.9,0.95,0.99,1.0]))
    print(name, 'NN dist hist', np.histogram(nn, bins=[0,1e-6,1e-3,0.01,0.1,0.5,1,2,5,50])[0])
