from utils.Hicken_Method import Hicken_eval, Hicken_deriv_eval
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

fig4_data = {}

sns.set(style='ticks')

def set_fonts():
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=16)    # legend fontsize
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
set_fonts()

Func = pickle.load( open( "SAVED_DATA/_Saved_Circles.pkl", "rb" ) )

x = [-18.,18.]
y = [-9,6]

xticks = []
yticks = [0]
fig = plt.figure(figsize=(8,7), dpi=160)
ax1 = plt.subplot(2,1,1)
res = 500
xx,yy = np.meshgrid(np.linspace(x[0],x[1],res),
                    np.linspace(y[0],y[1],res))
pts = np.stack((xx.flatten(),yy.flatten()),axis=1)
phi = Hicken_eval(pts,KDTree(Func.exact[0]),Func.exact[1],15,10)
phi = phi.reshape(res,res)
# print(np.min(phi))
# print(np.max(phi))
fig4_data["pt_cloud"] = Func.exact[0]
fig4_data["phi_exact_grid"] = phi.copy()
phi[phi<0] /= 10
phi[phi<0] = -1*(-phi[phi<0])**(0.7)
phi[phi>0] /= 3
phi[phi>0] = phi[phi>0]**(0.7)
shading = ax1.pcolormesh(xx,yy,phi,shading='gouraud',cmap='RdYlBu', vmin=-1, vmax=1)
for i in range(4):
    rng = np.arange(10000*i,10000*(i+1)-1)
    if i==0:
        ax1.plot(Func.exact[0][rng,0],Func.exact[0][rng,1],'k-',label='Boundary $\Gamma$',linewidth=2)
    else:
        ax1.plot(Func.exact[0][rng,0],Func.exact[0][rng,1],'k-',linewidth=2)
ax1.plot([x[0],x[1]],[0,0],color='cyan',linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
# ax1.set_title('Multi-Circle')
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.axis('equal')
ax1.set_xlim(-18,18)
# cax = ax1.inset_axes([0.6, 1.01, 0.4, 0.05])
# cbar = fig.colorbar(shading, cax=cax, ax=ax1, ticks=[], orientation='horizontal')
cax = ax1.inset_axes([1.01, 0.25, 0.04, 0.5])
cbar = fig.colorbar(shading, cax=cax, ax=ax1, ticks=[-1,0,1], orientation='vertical')
cbar.ax.set_title('$d_\Gamma$',fontsize=18)
cbar.ax.set_yticklabels(['$< 0$','$0$','$> 0$'], fontsize=18)
sns.despine()

ax2 = plt.subplot(2,1,2,sharex=ax1)
sns.set()
xspan = np.linspace(x[0],x[1],res)
yspan = np.zeros(res)
pts = np.stack((xspan,yspan),axis=1)
u,v = Func.spatial_to_parametric(pts)
b = Func.Surface.get_basis_matrix(u,v,0,0)
phi = b.dot(Func.cps[:,2])
fig4_data["phi_energy_minimized"] = phi
# b10 = Func.Surface.get_basis_matrix(u,v,1,0)
# dx = b10.dot(Func.cps[:,2])
# b01 = Func.Surface.get_basis_matrix(u,v,0,1)
# dy = b01.dot(Func.cps[:,2])
# phi_ex = np.linalg.norm(np.column_stack((dx,dy)),axis=1)

# num_samples = np.size(Func.exact[0],0)
# np.random.seed(1)
# rng = np.random.default_rng()
# indx = rng.choice(np.size(Func.exact[0],0), size=num_samples, replace=False)
# down_pts = Func.exact[0][indx,:]
# down_norms = Func.exact[1][indx,:]

phi_ex = Hicken_eval(pts,KDTree(Func.exact[0]),Func.exact[1],15,10)
fig4_data["phi_exact_line"] = phi_ex
# phi_ex = Hicken_deriv_eval(pts,KDTree(down_pts),down_norms,15,10)
# phi_ex = phi_ex[:,0]
min_rngs = [[-12,-7],[-6,0],[6,9]]
max_rngs = [[0,4]]
vals = np.empty((0,1))
locs = np.empty((0,1))
for rng in min_rngs:
    ind = xspan>rng[0]
    ind *= xspan<rng[1]
    vals = np.vstack((vals,np.min(phi_ex[ind])))
    ind = phi_ex==np.min(phi_ex[ind])
    locs = np.vstack((locs,xspan[ind]))
for rng in max_rngs:
    ind = xspan>rng[0]
    ind *= xspan<rng[1]
    vals = np.vstack((vals,np.max(phi_ex[ind])))
    ind = phi_ex==np.max(phi_ex[ind])
    locs = np.vstack((locs,xspan[ind]))  

fig4_data["non_differentiables"] = np.column_stack((locs,vals))

ax2.plot(xspan,phi,linewidth=2,label='Our function $\phi$')
ax2.plot(xspan,phi_ex,'--',linewidth=2,label='Exact SDF $d_\Gamma$')
ax2.plot(locs,vals,'.',color='tab:red',markersize=15,label='Non-differentiable points')
ax2.set_xlabel('x')
ax2.set_ylabel('$\phi$')
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
# ax2.set_xlim(x[0],x[1])
ax2.legend(loc='lower left',framealpha=1,edgecolor='black',facecolor='white', fontsize=16, bbox_to_anchor=(0.02, 0.005))

ax1.plot(locs,np.zeros(len(locs)),'.',color='tab:red',markersize=15,label='Non-differentiable points')
ax1.legend(loc='lower left',framealpha=1,edgecolor='black',facecolor='white', fontsize=16, bbox_to_anchor=(0.02, 0.005))
ax2.grid()
sns.despine()
plt.tight_layout()

set_fonts()

# plt.savefig('PDF_figures/multi_circles.pdf',bbox_inches='tight') 

pickle.dump(fig4_data, open("fig4_data.pkl","wb"))

plt.show()