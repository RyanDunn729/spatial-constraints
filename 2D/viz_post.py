import numpy as np
import seaborn as sns

import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from utils.Hicken_Kaur import Hicken_eval

def set_fonts():
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
sns.set_style('ticks')
set_fonts()

Func = pickle.load( open( "_Saved_Function.pkl", "rb" ) )

print('dimensions: ',Func.dimensions)
print('Bbox_diag: ',Func.Bbox_diag)
spacing = np.max(np.diff(Func.cps[:,0:2],axis=0),axis=0)/Func.Bbox_diag
print('Control Point spacing:\n',spacing,'=',np.linalg.norm(spacing))
num_surf_pts = Func.num_surf_pts
num_cps_pts  = Func.num_cps_pts
num_hess_pts = Func.num_hess_pts
print('Num_hess_pts: ', num_hess_pts)
print('num_cps: ',Func.num_cps,'=',np.product(Func.num_cps))
print('Num_surf_pts: ', num_surf_pts,'\n')

x = Func.dimensions[0]
y = Func.dimensions[1]
phi_cps = Func.cps[:,2]

fig1 = plt.figure()
ax = plt.axes()
res = 1000
uu,vv = np.meshgrid(np.linspace(0,1,res),
                    np.linspace(0,1,res))
b = Func.Surface.get_basis_matrix(uu.flatten(),vv.flatten(),0,0)
xx = b.dot(Func.cps[:,0]).reshape(res,res)
yy = b.dot(Func.cps[:,1]).reshape(res,res)
phi = b.dot(phi_cps).reshape(res,res)
CS = ax.contour(xx,yy,phi,linestyles='dashed',levels=[-2,-1,0,1,2],linewidths=2,
    colors=['tab:red','tab:orange','tab:green','tab:blue','tab:purple'])
ax.clabel(CS, CS.levels, inline=True, fontsize=14, fmt={-2:'-2',-1:'-1',0:'',1:'1',2:'2'},
    inline_spacing=14, rightside_up=True)
ax.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-',label='Boundary')
ax.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],'k.',label='Surface Points')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xticks(np.arange(-10,5,1))
ax.set_yticks(np.arange(-10,10,1))
# ax.set_xlim(Func.dimensions[0,0], Func.dimensions[0,1])
# ax.set_ylim(Func.dimensions[1,0], Func.dimensions[1,1])
ax.axis('equal')
sns.despine()
ax.legend(framealpha=1,edgecolor='black',facecolor='white')

sns.set_style('ticks')
set_fonts()

plt.figure()
ax = plt.axes(projection='3d')
res = 300
ax.plot(Func.cps[:,0],Func.cps[:,1],phi_cps,'k.')
uu,vv = np.meshgrid(np.linspace(0,1,res),
                    np.linspace(0,1,res))
b = Func.Surface.get_basis_matrix(uu.flatten(),vv.flatten(),0,0)
xx = b.dot(Func.cps[:,0]).reshape(res,res)
yy = b.dot(Func.cps[:,1]).reshape(res,res)
phi = b.dot(phi_cps).reshape(res,res)
ax.contour(xx, yy, phi, levels=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('$\Phi$')
ax.set_title('Control Points')
ax.set_xticks([x[0],np.sum(x)/2,x[1]])
ax.set_yticks([y[0],np.sum(y)/2,y[1]])
dx = np.diff(x)
dy = np.diff(y)
if dx > dy:
    lim = (np.min(Func.cps[:,0]),np.max(Func.cps[:,0]))
else:
    lim = (np.min(Func.cps[:,1]), np.max(Func.cps[:,1]))
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_zlim(-5,5)

sns.set_style('ticks')
set_fonts()


dataset=KDTree(Func.surf_pts)
normals=Func.normals
k=10
rho=10
plt.figure()
ax = plt.axes()
res = 200
ones = np.ones(res)
diag = np.linspace(0,1,res)
basis = Func.Surface.get_basis_matrix(diag, 0.5*ones, 0, 0)
pts = basis.dot(Func.cps[:,2])
ax.plot(diag, pts, '-', color='C1', label='X-axis')
xyz = basis.dot(Func.cps[:,0:2])
sdf = Hicken_eval(xyz,dataset,normals,k,rho)
ax.plot(diag, sdf, '--', color='C1')
basis = Func.Surface.get_basis_matrix(0.5*ones, diag, 0, 0)
pts = basis.dot(Func.cps[:,2])
ax.plot(diag, pts, '-', color='C2', label='Y-axis')
xyz = basis.dot(Func.cps[:,0:2])
sdf = Hicken_eval(xyz,dataset,normals,k,rho)
ax.plot(diag, sdf, '--', color='C2')
ax.axis([0,1,np.min(Func.cps[:,2]),np.max(Func.cps[:,2])])
ax.set_xticks([0,0.5,1])
ax.set_yticks([np.min(Func.cps[:,2]),0,np.max(Func.cps[:,2])])
ax.set_ylim(np.min(Func.cps[:,2]),np.max(Func.cps[:,2]))
ax.set_xlabel('Normalized Location')
ax.set_ylabel('Phi')
ax.set_title('Phi along 1D slices')
ax.legend()


sns.set_style('ticks')
set_fonts()

res = 45
bbox_max = 5

# data, err = Func.check_local_RMS_error(bbox_max,res)
# plt.figure()
# ax = plt.axes()
# ax.plot(data,err,label='test')
# ax.set_title('RMS error away from the surface')
# ax.set_xlabel('$\epsilon$')
# ax.set_ylabel('RMS error')
# plt.legend(loc='lower left')

# ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
# RMS_local = np.mean(data)
phi = Func.eval_surface()
MAX_surf = np.max(abs(phi))/Func.Bbox_diag
RMS_surf = np.sqrt(np.sum(phi**2)/len(phi))/Func.Bbox_diag

dx,dy = Func.gradient_eval_surface()
nx,ny = Func.normals[:,0],Func.normals[:,1]
print("avg gradient error: ",np.mean( (dx+nx)**2 + (dy+ny)**2))

# print('RMS Local 1%:',RMS_local)
print('MAX surf:',MAX_surf)
print('RMS surf:',RMS_surf)

plt.show()