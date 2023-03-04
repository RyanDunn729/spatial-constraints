from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

### Load in data
data = pickle.load(open("fig3_data.pkl","rb"))
xx = data["xx"]
yy = data["yy"]
pt_cloud = data["pt_cloud"]
phi_init = data["phi_init"]
phi_energy_minimized = data["phi_energy_minimized"]

### Define exact SDF contours
box1 = np.array([[-2.5,-3.5],
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

### configurables
dimensions = np.array([[-4.,4.],
                       [-5.6,5.6]])
legend_anchor = (1.1, 1.05)
contour_alpha = 0.30
tickfontsize = 14
axisfontsize = 16

def set_fonts():
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=tickfontsize)    # legend fontsize
    plt.rc('axes', labelsize=axisfontsize)    # fontsize of the x and y labels

### loop over both figures
for i,phi in enumerate([phi_init, phi_energy_minimized]):
    sns.set_style('ticks')
    set_fonts()
    fig = plt.figure(figsize=(7,5),dpi=140)
    ax = plt.axes()
    ax.plot(box1[:,0],box1[:,1],'k-',label='Boundary $\Gamma$',linewidth=1)
    rect = patches.FancyBboxPatch((-3.5,-4.5), 7,9, boxstyle='round,pad=0,rounding_size=1', 
        linewidth=1.5, alpha=contour_alpha,edgecolor='k',facecolor='none')
    ax.add_patch(rect)
    ax.plot(box2[:,0],box2[:,1],'k-',alpha=contour_alpha,linewidth=1.5,label='Contours of $d_\Gamma$')
    ax.plot(box3[:,0],box3[:,1],'k-',alpha=contour_alpha,linewidth=1.5)
    ax.plot(pt_cloud[:,0],pt_cloud[:,1],'k.',markersize=5,label='Point cloud')
    CS = ax.contour(xx,yy,phi,linestyles='dashed',levels=[-1,0,1,2],linewidths=2,
        colors=['tab:orange','tab:green','tab:blue','tab:purple'])
    ax.clabel(CS, CS.levels, inline=True, fontsize=axisfontsize, fmt={-1:'-1',0:'0',1:'1',2:'2'},
        inline_spacing=18, rightside_up=True,
        manual=[(-4,0),(-3,0),(-2,0),(-1,0)])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks(np.arange(-4,5,1),fontsize=tickfontsize)
    plt.yticks(np.arange(-4,5,1),fontsize=tickfontsize)
    ax.set_xlim(1.25*dimensions[0,0], 2.5*dimensions[0,1])
    ax.set_ylim(dimensions[1,0], dimensions[1,1])

    sns.despine()

    axins = zoomed_inset_axes(ax, 5, loc=5)
    axins.plot(box1[:,0],box1[:,1],'k-',linewidth=1)
    axins.plot(pt_cloud[:,0],pt_cloud[:,1],'k.',markersize=10)
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
    
    if i==0:
        plt.savefig('Fig3a.pdf',bbox_inches='tight')
    if i==1:
        plt.savefig('Fig3b.pdf',bbox_inches='tight')

plt.show()
