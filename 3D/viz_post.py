import pickle
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import marching_cubes
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl.mesh import Mesh
from scipy.spatial import KDTree
from utils.Hicken_Kaur import Hicken_eval

def set_fonts():
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
sns.set_style('ticks')
set_fonts()

save_mesh = False
res = 180
isocontour = 0
mesh_name = f'Opt_Mesh_{isocontour}.stl'

size = (7.5,6.5)
dpi = 100

Func = pickle.load( open( "_Saved_Function.pkl", "rb" ) )

# Func = pickle.load( open( "SAVED_DATA/Opt_o4Bunny28_100000.pkl", "rb" ) )
# Func = pickle.load( open( "Opt_Bunny_For_OffSurface1.pkl", "rb" ) )
# Func = pickle.load( open( "SAVED_DATA/Opt_Heart_.pkl", "rb" ) )
# Func = pickle.load( open( "SAVED_DATA/Opt_Heart_.pkl", "rb" ) )

# Func = pickle.load( open( "SAVED_DATA/Opt_armadillo_.pkl", "rb" ) )
# mesh_name = 'Opt_armadillo_.stl'
# Func = pickle.load( open( "SAVED_DATA/Opt_buddha_.pkl", "rb" ) )
# mesh_name = 'Opt_buddha_.stl'
# Func = pickle.load( open( "SAVED_DATA/Opt_dragon_.pkl", "rb" ) )
# mesh_name = 'Opt_dragon_.stl'

# Func = pickle.load( open( "SAVED_DATA/Opt_Fuselage_.pkl", "rb" ) )
# Func = pickle.load( open( "SAVED_DATA/Opt_Human_.pkl", "rb" ) )
# Func = pickle.load( open( "SAVED_DATA/Opt_Luggage_.pkl", "rb" ) )
# Func = pickle.load( open( "SAVED_DATA/Opt_Wing_.pkl", "rb" ) )
# Func = pickle.load( open( "SAVED_DATA/Opt_Battery_.pkl", "rb" ) )
# Func = pickle.load( open( "SAVED_DATA/Opt_Heart_.pkl", "rb" ) )
print('dimensions: ',Func.dimensions)
print('Bbox_diag: ',Func.Bbox_diag)
spacing = np.max(np.diff(Func.cps[:,0:3],axis=0),axis=0)/Func.Bbox_diag
print('Control Point spacing:\n',spacing,'=',np.linalg.norm(spacing))

ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
print('epsilon error: ',np.mean(data))
num_surf_pts = Func.num_surf_pts
num_cps_pts  = Func.num_cps_pts
num_hess_pts = Func.num_hess_pts
print('Num_hess_pts: ', num_hess_pts)
print('num_cps: ',Func.num_cps,'=',np.product(Func.num_cps))
print('Num_surf_pts: ', num_surf_pts,'\n')

gold = (198/255, 146/255, 20/255)

dataset=KDTree(Func.surf_pts)
normals=Func.normals
k=10
rho=10

plt.figure(figsize=size,dpi=dpi)
ax = plt.axes()
res = 200
ones = np.ones(res)
diag = np.linspace(0,1,res)
basis = Func.Volume.get_basis_matrix(diag, 0.5*ones, 0.5*ones, 0, 0, 0)
pts = basis.dot(Func.cps[:,3])
ax.plot(diag, pts, '-', color='C1', label='X-axis')
xyz = basis.dot(Func.cps[:,0:3])
sdf = Hicken_eval(xyz,dataset,normals,k,rho)
ax.plot(diag, sdf, '--', color='C1')
basis = Func.Volume.get_basis_matrix(0.5*ones, diag, 0.5*ones, 0, 0, 0)
pts = basis.dot(Func.cps[:,3])
ax.plot(diag, pts, '-', color='C2', label='Y-axis')
xyz = basis.dot(Func.cps[:,0:3])
sdf = Hicken_eval(xyz,dataset,normals,k,rho)
ax.plot(diag, sdf, '--', color='C2')
basis = Func.Volume.get_basis_matrix(0.5*ones, 0.5*ones, diag, 0, 0, 0)
pts = basis.dot(Func.cps[:,3])
ax.plot(diag, pts, '-', color='C3', label='Z-axis')
xyz = basis.dot(Func.cps[:,0:3])
sdf = Hicken_eval(xyz,dataset,normals,k,rho)
ax.plot(diag, sdf, '--', color='C3')
ax.axis([0,1,np.min(Func.cps[:,3]),np.max(Func.cps[:,3])])
ax.set_xticks([0,0.5,1])
ax.set_yticks([np.min(Func.cps[:,3]),0,np.max(Func.cps[:,3])])
ax.set_ylim(np.min(Func.cps[:,3]),np.max(Func.cps[:,3]))
ax.set_xlabel('Normalized Location')
ax.set_ylabel('Phi')
ax.set_title('Phi along 1D slices')
ax.legend()

############ Plot pts > 0 #############
res = 45
u = np.einsum('i,j,k->ijk', np.linspace(0, 1, res), np.ones(res),np.ones(res)).flatten()
v = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(0, 1, res),np.ones(res)).flatten()
w = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(0, 1, res)).flatten()
pts = Func.Volume.evaluate_points(u, v, w)
fig = plt.figure(figsize=size,dpi=dpi)
ax = fig.add_subplot(111, projection='3d')
plt.plot(pts[pts[:,3]>0,0],pts[pts[:,3]>0,1],pts[pts[:,3]>0,2],'k.')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title('Pts > 0')
center = np.mean(Func.dimensions,axis=1)
d = np.max(np.diff(Func.dimensions,axis=1))
ax.set_xlim(center[0]-d/2, center[0]+d/2)
ax.set_ylim(center[1]-d/2, center[1]+d/2)
ax.set_zlim(center[2]-d/2, center[2]+d/2)

############ Local Error #############
# res = 2
# bbox_perc = 100*5/Func.Bbox_diag

# ep_max = bbox_perc*Func.Bbox_diag / 100
# ep_range = np.linspace(-ep_max,ep_max,res)
# # dataset = KDTree(Func.exact[0])
# RMS_local = np.zeros(len(ep_range))
# sample_pts = Func.surf_pts
# sample_normals = Func.normals
# dataset = KDTree(sample_pts)
# for i,ep in enumerate(ep_range):
#         i_pts = Func.exact[0] + ep*Func.exact[1]
#         phi = KS_eval(i_pts,dataset,sample_normals,10,10)
#         phi_exact,_ = dataset.query(i_pts,k=1)
#         RMS_local[i] = np.sqrt(np.mean(  (abs(phi)-phi_exact)**2  ))
# KS_error = RMS_local/Func.Bbox_diag

# ep_range,data = Func.check_local_RMS_error(bbox_perc,res,num_samp=20000)

# sns.set(style='ticks')
# plt.figure(figsize=(6,5),dpi=180)
# ax = plt.axes()
# plt.semilogy(ep_range/100,data,'-',color='tab:blue',label='Our Method')
# # plt.semilogy(ep_range/100,KS_error,'-',color='tab:orange',label='Explicit Method')
# ax.set_xlabel("Normalized Outward Distance",fontsize=14)
# ax.set_ylabel("RMS Error",fontsize=14)
# ax.set_title('Local distance error')
# ax.set_xlim(-bbox_perc/100,bbox_perc/100)
# ax.legend(loc='upper center',fontsize=12)
# ax.grid()
# sns.despine()
# plt.tight_layout()

# plt.savefig('PDF_figures/Err_v_ep.pdf',bbox_inches='tight')

plt.figure(figsize=size,dpi=dpi)
res = 50
ax = plt.axes(projection='3d')
x = Func.dimensions[0]
y = Func.dimensions[1]
z = Func.dimensions[2]
u = np.einsum('i,j,k->ijk', np.linspace(0,1,res), np.ones(res),np.ones(res)).flatten()
v = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(0,1,res),np.ones(res)).flatten()
w = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(0,1,res)).flatten()
basis = Func.Volume.get_basis_matrix(u, v, w, 0, 0, 0)
phi = basis.dot(Func.cps[:,3]).reshape((res,res,res))
verts, faces,_,_ = marching_cubes(phi, 0)
verts = verts*np.diff(Func.dimensions).flatten()/(res-1) + Func.dimensions[:,0]
level_set = Poly3DCollection(verts[faces],linewidth=0.25,alpha=1,facecolor=gold,edgecolor='k')
ax.add_collection3d(level_set)
# ax.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],Func.surf_pts[:,2],
#         'k.',label='surface points')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title('Current Level Set $n_{\Gamma}$=%i'%len(Func.surf_pts))
ax.set_xticks([x[0],(x[1]+x[0])/2,x[1]])
ax.set_yticks([y[0],(y[1]+y[0])/2,y[1]])
ax.set_zticks([z[0],(z[1]+z[0])/2,z[1]])
center = np.mean(Func.dimensions,axis=1)
d = np.max(np.diff(Func.dimensions,axis=1))
ax.set_xlim(center[0]-d/2, center[0]+d/2)
ax.set_ylim(center[1]-d/2, center[1]+d/2)
ax.set_zlim(center[2]-d/2, center[2]+d/2)

if save_mesh:
    x = Func.dimensions[0]
    y = Func.dimensions[1]
    z = Func.dimensions[2]
    u = np.einsum('i,j,k->ijk', np.linspace(0,1,res), np.ones(res),np.ones(res)).flatten()
    v = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(0,1,res),np.ones(res)).flatten()
    w = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(0,1,res)).flatten()
    basis = Func.Volume.get_basis_matrix(u, v, w, 0, 0, 0)
    phi = basis.dot(Func.cps[:,3]).reshape((res,res,res))
    verts, faces,_,_ = marching_cubes(phi,isocontour)
    verts = verts*np.diff(Func.dimensions).flatten()/(res-1) + Func.dimensions[:,0]
    surf = Mesh(np.zeros(faces.shape[0], dtype=Mesh.dtype))
    for i, f in enumerate(faces):
            for j in range(3):
                    surf.vectors[i][j] = verts[f[j],:]
    surf.save(mesh_name)
    print('Number of Verices: ',len(verts))

dx,dy,dz = Func.gradient_eval_surface()
nx,ny,nz = Func.normals[:,1],Func.normals[:,1],Func.normals[:,2]
print("normal vec error: ",np.sum( (dx+nx)**2 + (dy+ny)**2 + (dz+nz)**2)/num_surf_pts)

phi = Func.eval_surface()
phi = phi/Func.Bbox_diag
print('phi0_min: ',np.min(Func.cps[:,3]))
print('phi0_max: ',np.max(Func.cps[:,3]),'\n')
print('Surface error (relative): \n',
        'Max: ',np.max(phi),'\n',
        'RMS: ',np.sqrt(np.sum(phi**2)/len(phi)))
print('Diagonal: ',Func.Bbox_diag)
print('Surface error (units): \n',
        'Max: ',Func.Bbox_diag*np.max(phi),'\n',
        'RMS: ',Func.Bbox_diag*np.sqrt(np.sum(phi**2)/len(phi)))
print("Runtime: ", Func.runtime)
print("Diagonal: ", Func.Bbox_diag)
plt.show()