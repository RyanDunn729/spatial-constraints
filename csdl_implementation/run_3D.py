from models.EnergyMinProblem import EnergyMinProblem
from geom_shapes.ellipsoid import Ellipsoid
from models.base_3D import MyProblem
from utils.read_stl import extract_stl_info
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
dim = 3
order = 4
border = 0.15

Lp = 1e3  # Surface weighting
Ln = 1e1  # Gradient weighting
Lr = 1e-1  # Curvature weighting

# Include an extremely high res sample if needed
exact_filename = None

### BUNNY ###
max_cps = 28
flag = 'Bunny'
tol = 5e-4
filename = 'geom_shapes/Bunny_9000.stl'

### Dragon ###
# max_cps = 35
# flag = 'Dragon'
# tol = 1e-4
# filename = 'geom_shapes/dragon_100k.stl'

### Armadillo ###
# max_cps = 31
# flag = 'Armadillo'
# tol = 1e-4
# filename = 'geom_shapes/armadillo_100k.stl'

### Buddha ###
# max_cps = 37
# flag = 'Buddha'
# tol = 1e-4
# filename = 'geom_shapes/buddha_100k.stl'

### Ellipsoid ###
# max_cps = 42
# flag = 'Ellipsoid'
# tol = 1e-5
# num_pts = 500
# a = 8
# b = 7.25
# c = 5.75

visualize_init = False

### BUNNY varying Ngamma ###
# max_cps = 36
# flag = 'Bunny'
# tol = 5e-5
# filename = 'geom_shapes/Bunny_100000.stl'
# filename_exact = 'geom_shapes/Bunny.stl'     # 129951 vertices
# filename = 'geom_shapes/Bunny_500.stl'
# filename = 'geom_shapes/Bunny_808.stl'
# filename = 'geom_shapes/Bunny_1310.stl'
# filename = 'geom_shapes/Bunny_2120.stl'
# filename = 'geom_shapes/Bunny_3432.stl'
# filename = 'geom_shapes/Bunny_5555.stl'
# filename = 'geom_shapes/Bunny_9000.stl'
# filename = 'geom_shapes/Bunny_14560.stl'
# filename = 'geom_shapes/Bunny_25000.stl'
# filename = 'geom_shapes/Bunny_38160.stl'
# filename = 'geom_shapes/Bunny_64000.stl'
# filename = 'geom_shapes/Bunny_100000.stl'

######### Get Surface #########
if flag == 'Ellipsoid':
    num_pts = 500
    num_exact_pts = 10000
    a = 8
    b = 7.25
    c = 5.75
    e = Ellipsoid(a,b,c)
    exact = np.stack((e.points(num_exact_pts),e.unit_pt_normals(num_exact_pts)))
    surf_pts = e.points(num_pts)
    normals = e.unit_pt_normals(num_pts)
elif exact_filename is not None:
    surf_pts, normals = extract_stl_info(filename)
    exact = extract_stl_info(exact_filename)
else:
    surf_pts, normals = extract_stl_info(filename)
    exact = extract_stl_info("geom_shapes/{}_exact.stl".format(flag))

######### Initialize Volume #########
Func = MyProblem(surf_pts, normals, max_cps, border, order, exact=exact)
scaling, bases_surf, bases_curv = Func.get_values()
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
    Func.get_basis(loc='surf',du=1,dv=0,dw=0),
    Func.get_basis(loc='surf',du=0,dv=1,dw=0),
    Func.get_basis(loc='surf',du=0,dv=0,dw=1),
]
hessian_bases = [
    Func.get_basis(loc='hess',du=2,dv=0,dw=0),
    Func.get_basis(loc='hess',du=1,dv=1,dw=0),
    Func.get_basis(loc='hess',du=0,dv=2,dw=0),
    Func.get_basis(loc='hess',du=1,dv=0,dw=1),
    Func.get_basis(loc='hess',du=0,dv=1,dw=1),
    Func.get_basis(loc='hess',du=0,dv=0,dw=2),
]
#################################
model = csdl.Model()
model.create_input("phi_cps",shape=(num_cps_pts))
model.add(EnergyMinProblem(
    dim=dim,
    scaling=scaling,
    num_cps=num_cps_pts,
    N_gamma=num_surf_pts,
    N=num_hess_pts,
    Lp=Lp,
    Ln=Ln,
    Lr=Lr,
    scalar_basis=scalar_basis,
    gradient_bases=gradient_bases,
    hessian_bases=hessian_bases,
    normals=normals,
    bbox_diag=float(Func.Bbox_diag),
    verbose=True,
    dq=float(1/num_hess_pts),
))
model.add_design_variable("phi_cps", lower=-1, upper=1)
model.add_objective("objective", scaler=1)
#################################
sim = Simulator(model)
sim["phi_cps"] = Func.cps[:,3]
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
Func.set_cps(Func.Bbox_diag*sim['phi_cps'])
pickle.dump(Func, open( "_Saved_Function.pkl","wb"))
phi = Func.eval_surface()
phi = phi/Func.Bbox_diag
print('Surface error (rel): \n',
        'Max: ',np.max(abs(phi)),'\n',
        'RMS: ',np.sqrt(np.sum(phi**2)/len(phi)))
print('Surface error (units): \n',
        'Max: ',Func.Bbox_diag*np.max(abs(phi)),'\n',
        'RMS: ',Func.Bbox_diag*np.sqrt(np.mean(phi**2)))
print("Ep: ",sim["Ep"])
print("En: ",sim["En"])
print("Er: ",sim["Er"])
print('END')