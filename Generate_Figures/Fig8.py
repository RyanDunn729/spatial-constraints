import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

def set_fonts(legendfont=16,axesfont=16):
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=legendfont)    # legend fontsize
    plt.rc('axes', labelsize=axesfont)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=legendfont)
    plt.rc('ytick', labelsize=legendfont)
    return

data = pickle.load(open("fig8_data.pkl","rb"))
time_ours = data["time_ours"]
time_explicit = data["time_explicit"]
time_Lin_et_al = data["time_Lin_et_al"]
our_onsurf = data["our_onsurf"]
explicit_onsurf = data["explicit_onsurf"]
our_offsurf = data["our_offsurf"]
explicit_offsurf = data["explicit_offsurf"]
Ngamma_range = data["Ngamma_range"]

P_OM = np.polyfit(Ngamma_range,time_ours,1)
P_EM = np.polyfit(np.log(Ngamma_range),np.log(time_explicit),1)
P_CM = np.polyfit(np.log(Ngamma_range),np.log(time_Lin_et_al),1)

bf_OM = np.poly1d(P_OM)
bf_EM = np.poly1d(P_EM)
bf_CM = np.poly1d(P_CM)

sns.set(style='ticks')
set_fonts()
fig = plt.figure(figsize=(5,5),dpi=160)
ax1 = plt.axes()
ax1.loglog(Ngamma_range,bf_OM(Ngamma_range), 'k-',linewidth=6,alpha=0.13)
ax1.loglog(Ngamma_range,np.exp(bf_EM(np.log(Ngamma_range))), 'k-',linewidth=6,alpha=0.13)
ax1.loglog(Ngamma_range,np.exp(bf_CM(np.log(Ngamma_range))), 'k-',linewidth=6,alpha=0.13)

ax1.loglog(Ngamma_range,time_ours,'.-',label=('Our method'),color='tab:blue',markersize=10,linewidth=2)
ax1.loglog(Ngamma_range,time_explicit,'.--',label=('Hicken \& Kaur'),color='tab:orange',markersize=10,linewidth=2)
ax1.loglog(Ngamma_range,time_Lin_et_al,'.:',label=('Lin et al.'),color='tab:green',markersize=10,linewidth=2)

plt.text(1.5e3,5e-4,'$\mathcal{O}(N_{\Gamma})$',fontsize=14)
plt.text(5e4,3.8e-6,'$\mathcal{O}(1)$', fontsize=14)

ax1.set_xlabel('$N_{\Gamma}$',fontsize=14)
ax1.set_ylabel('Evaluation time per point (sec)',fontsize=14)
ax1.legend(fontsize=14,framealpha=1,edgecolor='black',facecolor='white')
ax1.grid()
ax1.set_ylim(1e-6,2e-2)
sns.despine()
plt.tight_layout()

plt.savefig('Fig8a.pdf',bbox_inches='tight')

sns.set(style='ticks')
set_fonts()
fig1 = plt.figure(figsize=(5,5),dpi=160)
ax1 = plt.axes()
ax1.loglog(Ngamma_range,our_onsurf,'.-',color='tab:blue',markersize=14,linewidth=2,label=('Our method'))
ax1.loglog(Ngamma_range,explicit_onsurf,'.--',color='tab:orange',markersize=14,linewidth=2,label=('Hicken \& Kaur'))
ax1.set_xlabel('$N_{\Gamma}$',fontsize=14)
ax1.set_ylabel('On-surface RMS error',fontsize=14)
ax1.legend(fontsize=12,framealpha=1,edgecolor='black',facecolor='white')
ax1.set_ylim(1e-4,2e-2)
ax1.grid()
sns.despine()
plt.tight_layout()
plt.savefig('Fig8b.pdf',bbox_inches='tight')

sns.set(style='ticks')
set_fonts(legendfont=12,axesfont=18)
fig2 = plt.figure(figsize=(5,5),dpi=160)
ax2 = plt.axes()
styles_bspline = ['.-','s-']
styles_KSmethod = ['.--','s--']
for i,ep in enumerate([0.5, 1]):
    if i==0:
        ax2.loglog(Ngamma_range,our_offsurf[:,i],styles_bspline[i],markersize=14,linewidth=2,color='tab:blue',
            label=('Our method ($\pm${})'.format(ep/100)))
    elif i==1:
        ax2.loglog(Ngamma_range,our_offsurf[:,i],styles_bspline[i],markersize=7,linewidth=2,color='tab:blue',
            label=('Our method ($\pm${})'.format(ep/100)))
for i,ep in enumerate([0.5, 1]):
    if i==0:
        ax2.loglog(Ngamma_range,explicit_offsurf[:,i],styles_KSmethod[i],markersize=14,linewidth=2,color='tab:orange',
            label=('Hicken \& Kaur ($\pm${})'.format(ep/100)))
    elif i==1:
        ax2.loglog(Ngamma_range,explicit_offsurf[:,i],styles_KSmethod[i],markersize=7,linewidth=2,color='tab:orange',
            label=('Hicken \& Kaur ($\pm${})'.format(ep/100)))
ax2.set_xlabel('$N_{\Gamma}$',fontsize=14)
ax2.set_ylabel('Off-surface RMS error',fontsize=14)
ax2.set_ylim(1e-4,2e-2)
ax2.legend(fontsize=12,framealpha=1,edgecolor='black',facecolor='white')
ax2.grid()
sns.despine()
plt.tight_layout()
plt.savefig('Fig8c.pdf',bbox_inches='tight')

plt.show()