from modules.Hicken_Method import Hicken_eval, Hicken_deriv_eval
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

sns.set(style='ticks')

def set_fonts():
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=14)    # legend fontsize
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
set_fonts()

Func = pickle.load( open( "SAVED_DATA/_Saved_Multio4.pkl", "rb" ) )
gold = (198/255, 146/255, 20/255)
blue = (24/255, 43/255, 73/255)

x = Func.dimensions[0]
y = Func.dimensions[1]

x = [-18.,18.]
y = [-9,6]

xticks = [0]
yticks = [0]
fig = plt.figure(figsize=(8,8), dpi=120)
ax1 = plt.subplot(2,1,1)
res = 300
xx,yy = np.meshgrid(np.linspace(x[0],x[1],res),
                    np.linspace(y[0],y[1],res))
pts = np.stack((xx.flatten(),yy.flatten()),axis=1)
phi = Hicken_eval(pts,KDTree(Func.exact[0]),Func.exact[1],15,10)
phi = phi.reshape(res,res)
ax1.pcolormesh(xx,yy,phi,shading='gouraud',cmap='binary', vmin=-5, vmax=5)
for i in range(4):
    rng = np.arange(10000*i,10000*(i+1)-1)
    if i==0:
        ax1.plot(Func.exact[0][rng,0],Func.exact[0][rng,1],'k-',label='Boundaries $\Gamma$',linewidth=2)
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
sns.despine()

ax2 = plt.subplot(2,1,2,sharex=ax1)
sns.set()
res = int(500)
xspan = np.linspace(x[0],x[1],res)
yspan = np.zeros(res)
pts = np.stack((xspan,yspan),axis=1)
u,v = Func.spatial_to_parametric(pts)
b = Func.Surface.get_basis_matrix(u,v,0,0)
phi = b.dot(Func.cps[:,2])
# b10 = Func.Surface.get_basis_matrix(u,v,1,0)
# dx = b10.dot(Func.cps[:,2])
# b01 = Func.Surface.get_basis_matrix(u,v,0,1)
# dy = b01.dot(Func.cps[:,2])
# phi_ex = np.linalg.norm(np.column_stack((dx,dy)),axis=1)

num_samples = np.size(Func.exact[0],0)
np.random.seed(1)
rng = np.random.default_rng()
indx = rng.choice(np.size(Func.exact[0],0), size=num_samples, replace=False)
down_pts = Func.exact[0][indx,:]
down_norms = Func.exact[1][indx,:]

phi_ex = Hicken_eval(pts,KDTree(down_pts),down_norms,15,10)
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

ax2.plot(xspan,phi,linewidth=2,label='Our function')
ax2.plot(xspan,phi_ex,'--',linewidth=2,label='Exact signed distance')
ax2.plot(locs,vals,'.',color='tab:red',markersize=15,label='Non-differentiable points')
ax2.set_xlabel('x')
ax2.set_ylabel('$\phi$')
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
# ax2.set_xlim(x[0],x[1])
ax2.legend(loc='lower center',framealpha=1,edgecolor='black',facecolor='white', fontsize="x-large")

ax1.plot(locs,np.zeros(len(locs)),'.',color='tab:red',markersize=15,label='Non-differentiable points')
ax1.legend(loc='lower left',framealpha=1,edgecolor='black',facecolor='white', fontsize="x-large")
ax2.grid()
sns.despine()
plt.tight_layout()

set_fonts()

plt.savefig('PDF_figures/multi_circles.pdf',bbox_inches='tight') 
 
plt.show()