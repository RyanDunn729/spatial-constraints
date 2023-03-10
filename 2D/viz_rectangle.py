import numpy as np
import seaborn as sns

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from scipy.spatial import KDTree

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from utils.Hicken_Method import Hicken_eval

fig3_data = {}

sns.set()
tickfontsize = 14
axisfontsize = 16
def set_fonts():
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=tickfontsize)    # legend fontsize
    plt.rc('axes', labelsize=axisfontsize)    # fontsize of the x and y labels
set_fonts()

exact_box = np.array([[-2.5,-3.5],
                 [-2.5,3.5],
                 [2.5,3.5],
                 [2.5,-3.5],
                 [-2.5,-3.5],])
box2 = np.array([[-1.5,-2.5],
                 [-1.5,2.5],
                 [1.5,2.5],
                 [1.5,-2.5],
                 [-1.5,-2.5],])
box3 = np.array([[-0.5,-1.5],
                 [-0.5,1.5],
                 [0.5,1.5],
                 [0.5,-1.5],
                 [-0.5,-1.5],])

legend_anchor = (1.1, 1.05)
contour_alpha = 0.30

Func = pickle.load( open( "SAVED_DATA/_Saved_Rectangle.pkl", "rb" ) )

gold = (198/255, 146/255, 20/255)
blue = (24/255, 43/255, 73/255)

x = Func.dimensions[0]
y = Func.dimensions[1]
phi_cps = Func.cps[:,2]

sns.set_style('ticks')
fig1 = plt.figure(figsize=(7,5),dpi=140)
ax = plt.axes()
res = 500
uu,vv = np.meshgrid(np.linspace(0,1,res),
                    np.linspace(0,1,res))
b = Func.Surface.get_basis_matrix(uu.flatten(),vv.flatten(),0,0)
xx = b.dot(Func.cps[:,0]).reshape(res,res)
yy = b.dot(Func.cps[:,1]).reshape(res,res)
phi = b.dot(phi_cps).reshape(res,res)
fig3_data["xx"] = xx
fig3_data["yy"] = yy
fig3_data["phi_energy_minimized"] = phi
fig3_data["pt_cloud"] = Func.surf_pts
ax.plot(exact_box[:,0],exact_box[:,1],'k-',label='Boundary $\Gamma$',linewidth=1)
rect = patches.FancyBboxPatch((-3.5,-4.5), 7,9, boxstyle='round,pad=0,rounding_size=1', 
    linewidth=1.5, alpha=contour_alpha,edgecolor='k',facecolor='none')
ax.add_patch(rect)
ax.plot(box2[:,0],box2[:,1],'k-',alpha=contour_alpha,linewidth=1.5,label='Contours of $d_\Gamma$')
ax.plot(box3[:,0],box3[:,1],'k-',alpha=contour_alpha,linewidth=1.5)
ax.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],'k.',markersize=5,label='Point cloud')
CS = ax.contour(xx,yy,phi,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
    colors=['tab:orange','tab:green','tab:blue','tab:purple'])
ax.clabel(CS, CS.levels, inline=True, fontsize=axisfontsize, fmt={-1:'-1',0:'0',1:'1',2:'2'},
    inline_spacing=18, rightside_up=True,
    manual=[(-4,0),(-3,0),(-2,0),(-1,0)])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.xticks(np.arange(-4,5,1),fontsize=tickfontsize)
plt.yticks(np.arange(-4,5,1),fontsize=tickfontsize)
ax.set_xlim(1.25*Func.dimensions[0,0], 2.5*Func.dimensions[0,1])
ax.set_ylim(Func.dimensions[1,0], Func.dimensions[1,1])
sns.despine()

axins = zoomed_inset_axes(ax, 5, loc=5)
axins.plot(exact_box[:,0],exact_box[:,1],'k-',linewidth=0.75)
axins.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],'k.',markersize=10)
axins.contour(xx,yy,phi,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
    colors=['tab:orange','tab:green','tab:blue','tab:purple'])
axins.set_xlim(2, 2.75)
axins.set_ylim(-3.75, -3)
axins.yaxis.tick_right()
axins.xaxis.tick_top()
plt.xticks(np.arange(2,3,0.25),fontsize=tickfontsize-2)
plt.yticks(np.arange(-3.75,-2.75,0.25),fontsize=tickfontsize-2)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='.2')
ax.legend(framealpha=1,edgecolor='black',facecolor='white',
            bbox_to_anchor=legend_anchor, loc='upper right')

set_fonts()
# plt.savefig('PDF_figures/Rectangle_Bspline.pdf',bbox_inches='tight')

sns.set_style('ticks')
fig2 = plt.figure(figsize=(7,5),dpi=140)
ax = plt.axes()
dataset = KDTree(Func.surf_pts)
samples = np.transpose(np.vstack((xx.flatten(),yy.flatten())))
phi_init = Hicken_eval(samples,dataset,Func.normals,k=6,rho=20)
phi_init = phi_init.reshape(res,res)
fig3_data["phi_init"] = phi_init
ax.plot(exact_box[:,0],exact_box[:,1],'k-',label='Boundary $\Gamma$',linewidth=1)
rect = patches.FancyBboxPatch((-3.5,-4.5), 7,9, boxstyle='round,pad=0,rounding_size=1', 
    linewidth=1.5, alpha=contour_alpha,edgecolor='k',facecolor='none')
ax.add_patch(rect)
ax.plot(box2[:,0],box2[:,1],'k-',alpha=contour_alpha,linewidth=1.5,label='Contours of $d_\Gamma$')
ax.plot(box3[:,0],box3[:,1],'k-',alpha=contour_alpha,linewidth=1.5)
ax.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],'k.',markersize=5,label='Point cloud')
CS = ax.contour(xx,yy,phi_init,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
    colors=['tab:orange','tab:green','tab:blue','tab:purple'])
ax.clabel(CS, CS.levels, inline=True, fontsize=axisfontsize, fmt={-1:'-1',0:'0',1:'1',2:'2'},
    inline_spacing=18, rightside_up=True,
    manual=[(-4,0),(-3,0),(-2,0),(-1,0)])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.xticks(np.arange(-4,5,1),fontsize=tickfontsize)
plt.yticks(np.arange(-4,5,1),fontsize=tickfontsize)
ax.set_xlim(1.25*Func.dimensions[0,0], 2.5*Func.dimensions[0,1])
ax.set_ylim(Func.dimensions[1,0], Func.dimensions[1,1])
sns.despine()

axins = zoomed_inset_axes(ax, 5, loc=5)
axins.plot(exact_box[:,0],exact_box[:,1],'k-',linewidth=1)
axins.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],'k.',markersize=10)
axins.contour(xx,yy,phi_init,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
    colors=['tab:orange','tab:green','tab:blue','tab:purple'])
axins.set_xlim(2, 2.75)
axins.set_ylim(-3.75, -3)
axins.yaxis.tick_right()
axins.xaxis.tick_top()
plt.xticks(np.arange(2,3,0.25),fontsize=tickfontsize-2)
plt.yticks(np.arange(-3.75,-2.75,0.25),fontsize=tickfontsize-2)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='.2')
ax.legend(framealpha=1,edgecolor='black',facecolor='white',
            bbox_to_anchor=legend_anchor, loc='upper right')

set_fonts()

# pickle.dump(fig3_data, open("fig3_data.pkl","wb"))
# plt.savefig('PDF_figures/Rectangle_Hicken.pdf',bbox_inches='tight')
plt.show()
