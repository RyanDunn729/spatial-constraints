from geom_shapes.multi_circle import multi_circle
from geom_shapes.multi_obj import multi_obj
from geom_shapes.rectangle import rectangle
from geom_shapes.ellipse import ellipse
from models.base import MyProblem
# from lsdo_viz.api import Problem
import matplotlib.pyplot as plt
import openmdao.api as om
import numpy as np
import pickle
import time
print('Imported Packages \n')

######### Configurables #########
dim = 2
order = 4
max_cps = 46

Lp = 1e0
Ln = 1e0
Lr = 1e-2

visualize_init = False

prev_filename = None
# prev_filename = "SAVED_DATA/Opt_rectangle_L2_1.pkl"
# prev_filename = "_Saved_Function.pkl"

######### Sample Surface #########
num_surf_pts = 76

a = 5
b = 7
e = rectangle(a,b)
custom_dimensions = np.array([
    [-4.,4.],
    [-5.6,5.6]])

# centers = [[-13.,-0.5],[-7.,2.],[2.,0.],[10.,-4.]]
# radii = [2.,2.,4.,3.]
# e = multi_circle(centers,radii)
# custom_dimensions = np.array([
#     [-18.,18.],
#     [-9,6]])

num_exact = 10000
pts = e.points(num_surf_pts)
normals = e.unit_pt_normals(num_surf_pts)
ex_pts = e.points(num_exact)
ex_norms = e.unit_pt_normals(num_exact)
exact = np.stack((ex_pts,ex_norms))

### WFLOP offshore example
# exact = pickle.load(open("geom_shapes/WFLOP_boundary_data.pkl","rb"))
# pts, normals = exact
# custom_dimensions = np.array([[0.,12.],
#                               [0.,12.],])

dxy = np.diff(custom_dimensions).flatten()
frac = dxy / np.max(dxy)
num_cps = np.zeros(2,dtype=int)
for i,ratio in enumerate(frac):
    if ratio < 0.75:
        ratio = 0.75
    num_cps[i] = int(frac[i]*max_cps)
    # num_cps[i] = int((frac[i]*max_cps)+order-1)
    num_cps[i] = 3*int((frac[i]*max_cps)/3)

######### Initialize Volume #########
Func = MyProblem(pts, normals, num_cps, order, custom_dimensions, exact=exact)
bbox_diag = Func.Bbox_diag
scaling = Func.scaling
phi_init = Func.cps[:,2]
# Key vector sizes
num_cps_pts  = Func.num_cps_pts
num_hess_pts = Func.num_hess_pts
num_surf_pts = Func.num_surf_pts
print('Num_hess_pts: ', num_hess_pts)
print('Num_ctrl_pts: ', num_cps_pts)
print('Num_surf_pts: ', num_surf_pts,'\n')
if visualize_init:
    Func.visualize_current(res=30)
    plt.show()
    exit()
#################################
A0 = Func.get_basis(loc='surf',du=0,dv=0).toarray()
Ax = scaling[0]*Func.get_basis(loc='surf',du=1,dv=0).toarray()
Ay = scaling[1]*Func.get_basis(loc='surf',du=0,dv=1).toarray()
Axx = scaling[0]*scaling[0]*Func.get_basis(loc='hess',du=2,dv=0).toarray()
Axy = scaling[0]*scaling[1]*Func.get_basis(loc='hess',du=1,dv=1).toarray()
Ayy = scaling[1]*scaling[1]*Func.get_basis(loc='hess',du=0,dv=2).toarray()

from numpy import newaxis as na
nx = normals[na,:,0]
ny = normals[na,:,1]

An = np.transpose(Ax)@Ax + np.transpose(Ay)@Ay
Ar = np.transpose(Axx)@Axx + 2*np.transpose(Axy)@Axy + np.transpose(Ayy)@Ayy

A = Lp/num_surf_pts * (np.transpose(A0)@A0)
A += Ln/num_surf_pts * (An)
A += Lr/num_hess_pts * (Ar)

b = Ln/num_surf_pts * (nx@Ax + ny@Ay)
phi_QP = np.linalg.solve(A,-b.flatten())
#################################
Func.set_cps(phi_QP)
pickle.dump(Func, open( "_Saved_Function.pkl","wb"))
phi = Func.eval_surface()
phi = phi/Func.Bbox_diag
print('Surface error (rel): \n',
        'Max: ',np.max(abs(phi)),'\n',
        'RMS: ',np.sqrt(np.sum(phi**2)/len(phi)))
print('Surface error (units): \n',
        'Max: ',Func.Bbox_diag*np.max(abs(phi)),'\n',
        'RMS: ',Func.Bbox_diag*np.sqrt(np.mean(phi**2)))
dx,dy = Func.gradient_eval_surface()
nx,ny = Func.normals[:,0],Func.normals[:,1]
print("avg gradient error: ",np.mean( (dx+nx)**2 + (dy+ny)**2))
print('END')