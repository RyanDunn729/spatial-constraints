from modules.objective import Objective
from modules.curv_samp import curv_sampling
from modules.surf_samp import surf_sampling
from geom_shapes.ellipsoid import Ellipsoid
from utils.read_stl import extract_stl_info
from modules.base import MyProblem
from modules.fnorm import Fnorm
import matplotlib.pyplot as plt
import openmdao.api as om
# from lsdo_viz.api import Problem
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
Lr = 1e-1 # Curvature weighting

# Include an extremely high res sample if needed
exact_filename = None

### BUNNY ###
max_cps = 32
flag = 'Bunny'
tol = 5e-4
filename = 'geom_shapes/Bunny_9000.stl'

### Fuselage ###
# max_cps = 44
# flag = 'Fuselage'
# tol = 1e-4
# filename = 'geom_shapes/Fuselage_25k.stl'

### Human ###
# max_cps = 34
# flag = 'Human'
# tol = 5e-4
# filename = 'geom_shapes/Human_25k.stl'

### Battery ###
# max_cps = 44
# flag = 'Battery'
# tol = 5e-4
# filename = 'geom_shapes/Battery_25k.stl'

### Luggage ###
# max_cps = 41
# flag = 'Luggage'
# tol = 1e-4
# filename = 'geom_shapes/Luggage_25k.stl'

### Wing ###
# max_cps = 44
# flag = 'Wing'
# tol = 5e-4
# filename = 'geom_shapes/Wing_25k.stl'

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

## HEART For Optimization ###
# max_cps = 34
# flag = 'Heart'
# tol = 1e-4
# filename = "geom_shapes/heart_case03_final1.stl"
# exact_filename = "geom_shapes/heart_case03_final1.stl"
# filename = "geom_shapes/heart_case02_v7.stl"

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
phi_init = Func.cps[:,3]
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
basis_200 = Func.get_basis('hess',du=2,dv=0,dw=0)
basis_110 = Func.get_basis('hess',du=1,dv=1,dw=0)
basis_020 = Func.get_basis('hess',du=0,dv=2,dw=0)
basis_101 = Func.get_basis('hess',du=1,dv=0,dw=1)
basis_011 = Func.get_basis('hess',du=0,dv=1,dw=1)
basis_002 = Func.get_basis('hess',du=0,dv=0,dw=2) 
dxx = scaling[0]*scaling[0]*basis_200.dot(phi_init)
dxy = scaling[0]*scaling[1]*basis_110.dot(phi_init)
dyy = scaling[1]*scaling[1]*basis_020.dot(phi_init)
dxz = scaling[0]*scaling[2]*basis_101.dot(phi_init)
dyz = scaling[1]*scaling[2]*basis_011.dot(phi_init)
dzz = scaling[2]*scaling[2]*basis_002.dot(phi_init)
Er0 = np.sum( dxx**2 + 2*dxy**2 + dyy**2 + 2*dxz**2 + 2*dyz**2 + dzz**2) / num_hess_pts
#################################
EnergyMinModel = om.Group()
EnergyMinModel.add_subsystem('Curvature_Sampling', curv_sampling(
        num_cps=num_cps_pts,
        num_pts=num_hess_pts,
        dim=dim,
        scaling=scaling,
        bases=bases_curv,
        bbox_diag=float(Func.Bbox_diag),
    ),promotes=['*'])
EnergyMinModel.add_subsystem('Surface_Sampling',surf_sampling(
        num_cps=num_cps_pts,
        num_pts=num_surf_pts,
        dim=dim,
        scaling=scaling,
        bases=bases_surf,
        bbox_diag=float(Func.Bbox_diag),
    ),promotes=['*'])
EnergyMinModel.add_subsystem('Fnorms',Fnorm(
        num_pts=num_hess_pts,
        dim=dim,
    ),promotes=['*'])
EnergyMinModel.add_subsystem('Objective',Objective(
        num_samp=num_hess_pts,
        num_surf=num_surf_pts,
        dim=dim,
        Lp=Lp,
        Ln=Ln,
        Lr=Lr,
        normals=normals,
        verbose=True,
    ),promotes=['*'])
#################################
# Prob = Problem()
Prob = om.Problem(EnergyMinModel)
Prob.model.add_design_var('phi_cps',lower=-1,upper=1)
Prob.model.add_objective('objective',scaler=1)
#################################
Prob.driver = om.pyOptSparseDriver()
Prob.driver.options['optimizer'] = 'SNOPT'
Prob.driver.opt_settings['Major iterations limit'] = 10000
Prob.driver.opt_settings['Minor iterations limit'] = 10000
# Prob.driver.opt_settings['Verify level'] = 0
Prob.driver.opt_settings['Iterations limit'] = 150000
Prob.driver.opt_settings['Major optimality tolerance'] = tol
Prob.setup()
#################################
Prob['phi_cps'] = Func.cps[:,3]
#################################
Prob.run_model()
t1 = time.time()
Prob.run_driver()
t2 = time.time()
#################################
print('Runtime: ',t2-t1)
Func.runtime = t2-t1
print('Final Objective Value: ',Prob['objective'])
print('max phi_surf: ',np.max(abs(Prob['phi_surf'])))
Func.set_cps(Prob['phi_cps']*Func.Bbox_diag)
# pickle.dump(Func, open( "SAVED_DATA/Opt_{}_.pkl".format(flag),"wb"))
pickle.dump(Func, open( "_Saved_Function.pkl","wb"))
# pickle.dump(Func, open( "Opt_{}_For_OffSurface3.pkl".format(flag),"wb"))
# pickle.dump(Func, open( SAVE_NAME,"wb"))
phi = Func.eval_surface()
phi = phi/Func.Bbox_diag
print('Surface error (rel): \n',
        'Max: ',np.max(abs(phi)),'\n',
        'RMS: ',np.sqrt(np.sum(phi**2)/len(phi)))
print('Surface error (units): \n',
        'Max: ',Func.Bbox_diag*np.max(abs(phi)),'\n',
        'RMS: ',Func.Bbox_diag*np.sqrt(np.mean(phi**2)))
ep_range,local_err = Func.check_local_RMS_error(1,10)
print('local_RMS_error: \n',np.transpose(np.stack((ep_range,local_err),axis=0)))
print('END')