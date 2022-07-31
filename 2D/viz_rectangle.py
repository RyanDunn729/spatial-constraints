import numpy as np
import seaborn as sns

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from scipy.spatial import KDTree

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from modules.Hicken_Method import KS_eval

sns.set()
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels

Func = pickle.load( open( "_Saved_Function.pkl", "rb" ) )

gold = (198/255, 146/255, 20/255)
blue = (24/255, 43/255, 73/255)

x = Func.dimensions[0]
y = Func.dimensions[1]
phi_cps = Func.cps[:,2]

sns.set_style('ticks')
fig1 = plt.figure(figsize=(7,5),dpi=140)
ax = plt.axes()
res = 1000
uu,vv = np.meshgrid(np.linspace(0,1,res),
                    np.linspace(0,1,res))
b = Func.Surface.get_basis_matrix(uu.flatten(),vv.flatten(),0,0)
xx = b.dot(Func.cps[:,0]).reshape(res,res)
yy = b.dot(Func.cps[:,1]).reshape(res,res)
phi = b.dot(phi_cps).reshape(res,res)
CS = ax.contour(xx,yy,phi,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
    colors=['tab:orange','tab:green','tab:blue','tab:purple'])
ax.clabel(CS, CS.levels, inline=True, fontsize=14, fmt={-1:'-1',0:'',1:'1',2:'2'},
    inline_spacing=14, rightside_up=True)
ax.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-',label='Boundary')
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
rect = patches.FancyBboxPatch((-3.5,-4.5), 7,9, boxstyle='round,pad=0,rounding_size=1', 
    linewidth=2, alpha=0.25,edgecolor='k',facecolor='none')
ax.add_patch(rect)
ax.plot(box2[:,0],box2[:,1],'k-',alpha=0.35,linewidth=1.5,label='Exact SDF')
ax.plot(box3[:,0],box3[:,1],'k-',alpha=0.35,linewidth=1.5)
CS = ax.contour(xx,yy,phi,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
    colors=['tab:orange','tab:green','tab:blue','tab:purple'])
ax.clabel(CS, CS.levels, inline=True, fontsize=14, fmt={-1:'-1',0:'',1:'1',2:'2'},
    inline_spacing=14, rightside_up=True)
# ax.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-')
ax.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],'k.',label='Surface Points')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xticks(np.arange(-10,5,1))
ax.set_yticks(np.arange(-10,10,1))
ax.set_xlim(1.25*Func.dimensions[0,0], 2.5*Func.dimensions[0,1])
ax.set_ylim(Func.dimensions[1,0], Func.dimensions[1,1])
sns.despine()
axins = zoomed_inset_axes(ax, 5, loc=7)
axins.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-',linewidth=1)
axins.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],'k.',markersize=10)
axins.contour(xx,yy,phi,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
    colors=['tab:orange','tab:green','tab:blue','tab:purple'])
axins.set_xlim(2, 2.75)
axins.set_ylim(-3.75, -3)
axins.yaxis.tick_right()
axins.xaxis.tick_top()
plt.xticks(np.arange(2,3,0.25),fontsize=10)
plt.yticks(np.arange(-3.75,-2.75,0.25),fontsize=10)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='.2')
ax.legend(framealpha=1,edgecolor='black',facecolor='white')

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.savefig('PDF_figures/Rectangle_Bspline.pdf',bbox_inches='tight')

sns.set_style('ticks')
fig2 = plt.figure(figsize=(7,5),dpi=140)
ax = plt.axes()
dataset = KDTree(Func.surf_pts)
samples = np.transpose(np.vstack((xx.flatten(),yy.flatten())))
phi = KS_eval(samples,dataset,Func.normals,k=6,rho=10)
phi = phi.reshape(res,res)
CS = ax.contour(xx,yy,phi,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
    colors=['tab:orange','tab:green','tab:blue','tab:purple'])
ax.clabel(CS, CS.levels, inline=True, fontsize=14, fmt={-1:'-1',0:'',1:'1',2:'2'},
    inline_spacing=14, rightside_up=True)
ax.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-',label='Boundary')
rect = patches.FancyBboxPatch((-3.5,-4.5), 7,9, boxstyle='round,pad=0,rounding_size=1', 
    linewidth=2, alpha=0.25,edgecolor='k',facecolor='none')
ax.add_patch(rect)
ax.plot(box2[:,0],box2[:,1],'k-',alpha=0.35,linewidth=1.5,label='Exact SDF')
ax.plot(box3[:,0],box3[:,1],'k-',alpha=0.35,linewidth=1.5)
CS = ax.contour(xx,yy,phi,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
    colors=['tab:orange','tab:green','tab:blue','tab:purple'])
ax.clabel(CS, CS.levels, inline=True, fontsize=14, fmt={-1:'-1',0:'',1:'1',2:'2'},
    inline_spacing=14, rightside_up=True)
# ax.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-')
ax.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],'k.',label='Surface Points')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xticks(np.arange(-10,5,1))
ax.set_yticks(np.arange(-10,10,1))
ax.set_xlim(1.25*Func.dimensions[0,0], 2.5*Func.dimensions[0,1])
ax.set_ylim(Func.dimensions[1,0], Func.dimensions[1,1])
sns.despine()
axins = zoomed_inset_axes(ax, 5, loc=7)
axins.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-',linewidth=1)
axins.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],'k.',markersize=10)
axins.contour(xx,yy,phi,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
    colors=['tab:orange','tab:green','tab:blue','tab:purple'])
axins.set_xlim(2, 2.75)
axins.set_ylim(-3.75, -3)
axins.yaxis.tick_right()
axins.xaxis.tick_top()
plt.xticks(np.arange(2,3,0.25),fontsize=10)
plt.yticks(np.arange(-3.75,-2.75,0.25),fontsize=10)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='.2')
ax.legend(framealpha=1,edgecolor='black',facecolor='white')

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels

plt.savefig('PDF_figures/Rectangle_Hicken.pdf',bbox_inches='tight')
plt.show()
