import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

### Load in data
data = pickle.load(open("fig4_data.pkl","rb"))
pt_cloud = data["pt_cloud"]
phi_exact_grid = data["phi_exact_grid"]
phi_exact_line = data["phi_exact_line"]
phi_energy_minimized = data["phi_energy_minimized"]
nondiff = data["non_differentiables"]

sns.set(style='ticks')

def set_fonts():
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=16)    # legend fontsize
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
set_fonts()

x = [-18.,18.]
y = [-9,6]
res = 500

xticks = []
yticks = [0]
fig = plt.figure(figsize=(8,7), dpi=160)
ax1 = plt.subplot(2,1,1)
xx,yy = np.meshgrid(np.linspace(x[0],x[1],res),
                    np.linspace(y[0],y[1],res))
phi_scaled = phi_exact_grid.copy()
phi_scaled[phi_scaled<0] /= 10
phi_scaled[phi_scaled<0] = -1*(-phi_scaled[phi_scaled<0])**(0.7)
phi_scaled[phi_scaled>0] /= 3
phi_scaled[phi_scaled>0] = phi_scaled[phi_scaled>0]**(0.7)
shading = ax1.pcolormesh(xx,yy,phi_scaled,shading='gouraud',cmap='RdYlBu', vmin=-1, vmax=1)
for i in range(4):
    rng = np.arange(10000*i,10000*(i+1)-1)
    if i==0:
        ax1.plot(pt_cloud[rng,0],pt_cloud[rng,1],'k-',label='Boundary $\Gamma$',linewidth=2)
    else:
        ax1.plot(pt_cloud[rng,0],pt_cloud[rng,1],'k-',linewidth=2)
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
xspan = np.linspace(x[0],x[1],500)

ax2.plot(xspan,phi_energy_minimized,linewidth=2,label='Our function $\phi$')
ax2.plot(xspan,phi_exact_line,'--',linewidth=2,label='Exact SDF $d_\Gamma$')
ax2.plot(nondiff[:,0],nondiff[:,1],'.',color='tab:red',markersize=15,label='Non-differentiable points')
ax2.set_xlabel('x')
ax2.set_ylabel('$\phi$')
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
# ax2.set_xlim(x[0],x[1])
ax2.legend(loc='lower left',framealpha=1,edgecolor='black',facecolor='white', fontsize=16, bbox_to_anchor=(0.02, 0.005))

ax1.plot(nondiff[:,0],np.zeros(len(nondiff[:,0])),'.',color='tab:red',markersize=15,label='Non-differentiable points')
ax1.legend(loc='lower left',framealpha=1,edgecolor='black',facecolor='white', fontsize=16, bbox_to_anchor=(0.02, 0.005))
ax2.grid()
sns.despine()
plt.tight_layout()

set_fonts()

plt.savefig('Fig4.pdf',bbox_inches='tight') 

plt.show()