from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def size(m,n,dm,dn):
    return m*n/(m*(m-dm) + np.minimum(n - dn, m - dm) + n*(n-dn))

tol = 1E-4
m = 640
n = 960
N_points = 1000000
dm = np.linspace(0, m, int(np.sqrt(N_points))+1)[:-1]
dn = np.linspace(0, n, int(np.sqrt(N_points))+1)[:-1]
DM,DN = np.meshgrid(dm, dn)

S = size(m,n,DM,DN)
S2 = np.ones_like(S)

S_diff = np.abs(S - S2)
shapes = S_diff.shape
S_diff = S_diff.flatten()
args = np.where(S_diff <= tol)
S_diff = S.flatten()[args]
DM_diff = DM.flatten()[args]
DN_diff = DN.flatten()[args]

fig = plt.figure()
ax = fig.gca(projection="3d")
fig.set_size_inches(8, 6)
fig.tight_layout()
ax.plot(DM_diff, DN_diff, np.log(S_diff), "r-")
plt.legend(["$R = 1$"])
ax.plot_surface(DM, DN, np.log10(S), cmap = cm.magma, antialiased = True,
alpha = 0.75)
ax.set_xlabel("$\\Delta m$")
ax.set_ylabel("$\\Delta n$")
ax.set_zlabel(f"Compression Ratio $R$")
labels = np.zeros_like(ax.get_zticks())
zmin = np.log10(np.min(S))
zmax = np.log10(np.max(S))
zstep = (zmax - zmin)/(len(labels))
for n,z in enumerate(labels):
    labels[n] = f"{10**(n*zstep + zmin):.0f}"
ax.set_zticklabels(labels)
plt.savefig("comp_ratio.pdf", dpi = 250)
plt.close()
