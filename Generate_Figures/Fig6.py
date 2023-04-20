import matplotlib.pyplot as plt
import seaborn as sns
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
set_fonts()

data = pickle.load(open("fig6_data.pkl","rb"))
RMS_local_L1 = data["RMS_local_L1"]
RMS_surf_L1 = data["RMS_surf_L1"]
RMS_local_L2 = data["RMS_local_L2"]
RMS_surf_L2 = data["RMS_surf_L2"]
RMS_local_L3 = data["RMS_local_L3"]
RMS_surf_L3 = data["RMS_surf_L3"]
lambda_range = data["lambda_range"]

# sns.set(style='ticks')
# fig3= plt.figure(figsize=(5,4),dpi=180)
# ax3 = plt.axes()
# # ax3.loglog(lambda_range,max_Fnorm_L3,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
# ax3.loglog(lambda_range,RMS_local_L3,'D-',linewidth=2,markersize=6,color='tab:red',label='Off-Surface $(\pm 0.01)$')
# ax3.loglog(lambda_range,RMS_surf_L3, 'o-',linewidth=2,markersize=6,color='tab:blue',label='On-Surface')
# # ax3.loglog(lambda_range,Runtime_L3/Runtime_L3[4],'.--',markersize=8,label='Optimization time')
# # ax3.loglog(lambda_range,MAX_surf_L3,'.-',markersize=8,color='tab:cyan',label='Max Surface')
# ax3.set_xlabel('$\lambda_p$',fontsize=16)
# ax3.set_ylabel('RMS Error',fontsize=16)
# ax3.set_xticks(lambda_range)
# ax3.legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white',fontsize=12)
# ax3.set_ylim(3e-4,1e2)
# ax3.set_ylim(1e-4,1e0)
# ax3.grid()
# sns.despine()
# plt.tight_layout()
# set_fonts()
# plt.savefig('Fig6.pdf',bbox_inches='tight')

sns.set(style='ticks')
fig2= plt.figure(figsize=(5,4),dpi=180)
ax2 = plt.axes()
# ax2.loglog(lambda_range,max_Fnorm_L2,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
ax2.loglog(lambda_range,RMS_local_L2,'D-',linewidth=2,markersize=6,color='tab:red',label='Off-surface $(\pm 0.01)$')
ax2.loglog(lambda_range,RMS_surf_L2, 'o-',linewidth=2,markersize=6,color='tab:blue',label='On-surface')
# ax2.loglog(lambda_range,Runtime_L2/Runtime_L2[4],'.--',markersize=8,label='Optimization time')
# ax2.loglog(lambda_range,MAX_surf_L2,'.-',markersize=8,color='tab:cyan',label='Max Surface')
ax2.set_xlabel('$\lambda_n$',fontsize=16)
ax2.set_ylabel('RMS Error',fontsize=16)
ax2.set_xticks(lambda_range)
ax2.legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white',fontsize=12)
ax2.set_ylim(3e-4,1e2)
ax2.set_ylim(1e-4,1e0)
ax2.grid()
sns.despine()
plt.tight_layout()
set_fonts()
plt.savefig('Fig6a.pdf',bbox_inches='tight')

sns.set(style='ticks')
fig1= plt.figure(figsize=(5,4),dpi=180)
ax1 = plt.axes()
ax1.loglog(lambda_range,RMS_local_L1,'D-',linewidth=2,markersize=6,color='tab:red',label='Off-surface $(\pm 0.01)$')
ax1.loglog(lambda_range,RMS_surf_L1,'o-', linewidth=2,markersize=6,color='tab:blue',label='On-surface')
ax1.set_xlabel('$\lambda_r$',fontsize=16)
ax1.set_ylabel('RMS Error',fontsize=16)
ax1.set_xticks(lambda_range)
ax1.legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white',fontsize=12)
ax1.set_ylim(3e-4,1e2)
ax1.set_ylim(1e-4,1e0)
ax1.grid()
sns.despine()
plt.tight_layout()
set_fonts()
plt.savefig('Fig6b.pdf',bbox_inches='tight')

plt.show()