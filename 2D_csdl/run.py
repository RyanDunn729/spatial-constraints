from models.EnergyMinProblem import EnergyMinProblem
from geom_shapes.multi_circle import multi_circle
from geom_shapes.multi_obj import multi_obj
from geom_shapes.rectangle import rectangle
from geom_shapes.ellipse import ellipse
from Bsplines.base import MyProblem
import csdl
from python_csdl_backend import Simulator
from modopt.csdl_library import CSDLProblem
import sys
if "win" in sys.platform:
    from modopt.scipy_library import SLSQP
else:
    from modopt.snopt_library import SNOPT

import numpy as np
import pickle
import time
print('Imported Packages \n')

######### Configurables #########
order = 4
max_cps = 40

Lp = 1e3
Ln = 10.
Lr = 1e-1

tol = 1e-4

visualize_init = False
######### Get Contour #########
# 72 is even
# 76 is slightly off
num_surf_pts = 76
a = 5
b = 7
# centers = [[-13.,-0.5],[-7.,2.],[2.,0.],[10.,-4.]]
# radii = [2.,2.,4.,3.]

### Choose shape ###
# e = ellipse(a,b)
e = rectangle(a,b)
# e = multi_circle(centers,radii)
###############################
pts = e.points(num_surf_pts)
normals = e.unit_pt_normals(num_surf_pts)

num_exact = 10000
ex_pts = e.points(num_exact)
ex_norms = e.unit_pt_normals(num_exact)
exact = np.stack((ex_pts,ex_norms))

exact = pickle.load(open("boundary_data.pkl","rb"))
pts, normals = exact
x = np.asarray([262403., 262553., 262703., 262853., 263003., 263153., 263303.,
                263453., 263603., 263753., 263903., 264053., 264203., 264353.,
                264503., 264653., 264803., 264953., 265103., 265253.])
y = np.asarray([6504239., 6504389., 6504539., 6504689., 6504839., 6504989.,
                6505139., 6505289., 6505439., 6505589., 6505739., 6505889.,
                6506039., 6506189., 6506339., 6506489., 6506639., 6506789.,
                6506939., 6507089.])
x_min_d = x.min()
y_min_d = y.min()
x -= x_min_d
y -= y_min_d
x /= np.max(x)/10
y /= np.max(y)/10
custom_dimensions = np.array([[x.min(), x.max()],
                              [y.min(), y.max()],])

dxy = np.diff(custom_dimensions).flatten()
frac = dxy / np.max(dxy)
num_cps = np.zeros(2,dtype=int)
for i,ratio in enumerate(frac):
    if ratio < 0.75:
        ratio = 0.75
    num_cps[i] = int(np.round(frac[i]*max_cps)+order-1)

######### Initialize Volume #########
Func = MyProblem(pts, normals, num_cps, order, custom_dimensions, exact=None)
scaling = Func.scaling
dA = Func.dA/Func.area
phi_init = Func.cps[:,2]
# Key vector sizes
num_surf_pts = Func.num_surf_pts
num_cps_pts  = Func.num_cps_pts
num_hess_pts = Func.num_hess_pts
print('Num_hess_pts: ', num_hess_pts)
print('Num_ctrl_pts: ', num_cps_pts)
print('Num_surf_pts: ', num_surf_pts,'\n')
#################################
scalar_basis = Func.get_basis(loc='surf',du=0,dv=0)
gradient_bases = [
    Func.get_basis(loc='surf',du=1,dv=0),
    Func.get_basis(loc='surf',du=0,dv=1),
]
hessian_bases = [
    Func.get_basis(loc='hess',du=2,dv=0),
    Func.get_basis(loc='hess',du=1,dv=1),
    Func.get_basis(loc='hess',du=0,dv=2),
]
#################################
model = csdl.Model()
model.create_input("phi_cps",shape=(num_cps_pts))
model.add(EnergyMinProblem(
    dim=2,
    scaling=scaling,
    num_cps=num_cps_pts,
    N_gamma=num_surf_pts,
    N=num_hess_pts,
    Lp=Lp,
    Ln=Ln,
    Lr=Lr,
    dV=dA,
    scalar_basis=scalar_basis,
    gradient_bases=gradient_bases,
    hessian_bases=hessian_bases,
    normals=normals/Func.Bbox_diag,
))
model.add_design_variable("phi_cps", lower=-1, upper=1)
model.add_objective("objective", scaler=1)
#################################
sim = Simulator(model)
sim["phi_cps"] = phi_init
sim.run()
#################################
prob = CSDLProblem(
    problem_name="opt",
    simulator=sim,
)
if "win" in sys.platform:
    optimizer = SLSQP(prob)
else:
    optimizer = SNOPT(
        prob,
        Major_iterations=10000,
        Major_optimality=tol,
        append2file=True,
    )
# Solve
t1 = time.time()
optimizer.solve()
t2 = time.time()
#################################
print('Runtime: ',t2-t1)
print('Final Objective Value: ',sim["objective"])
Func.runtime = t2-t1
Func.set_cps(sim['phi_cps']*Func.Bbox_diag)
pickle.dump(Func, open( "_Saved_Function.pkl","wb"))
phi = Func.eval_surface()
phi = phi/Func.Bbox_diag
print('Surface error (rel): \n',
        'Max: ',np.max(phi),'\n',
        'RMS: ',np.sqrt(np.sum(phi**2)/len(phi)))
print('Surface error (units): \n',
        'Max: ',Func.Bbox_diag*np.max(phi),'\n',
        'RMS: ',Func.Bbox_diag*np.sqrt(np.mean(phi**2)))
print('END')