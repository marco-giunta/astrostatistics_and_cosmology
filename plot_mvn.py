import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/ 
# nella variante dove np.einsum viene rimpiazzato da scipy.stats.multivariate_normal (vedi fine pagina)

N = 100
X = np.linspace(-1, 9, N)
Y = np.linspace(0, 4, N)
X, Y = np.meshgrid(X, Y)

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

mu = np.array([4, 2])
Sigma = np.array([[ 1.44, -0.702], [-0.702,  0.81]])

F0 = multivariate_normal(mu, Sigma)
Z0 = F0.pdf(pos)

k = 25
F = multivariate_normal(mu, Sigma/k)
Z = F.pdf(pos)

plt.contour(X, Y, Z0, colors = 'blue') #, label = 'original cov' # https://stackoverflow.com/questions/10490302/how-do-you-create-a-legend-for-a-contour-plot-in-matplotlib complicato...
plt.contour(X, Y, Z, colors = 'red') #, label = f'cov/{k}'
# plt.legend()
plt.title(f'blue originale, rosso cov/{k}') # soluzione scema
plt.show()