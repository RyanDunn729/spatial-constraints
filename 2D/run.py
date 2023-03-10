from models.objective import Objective
from models.surf_samp import surf_sampling
from models.curv_samp import curv_sampling
from models.fnorm import Fnorm
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
max_cps = 72

Lp = 1e3
Ln = 10.
Lr = 1e1

tol = 1e-4

visualize_init = False

prev_filename = None
# prev_filename = "SAVED_DATA/Opt_rectangle_L2_1.pkl"
# prev_filename = "_Saved_Function.pkl"

######### Sample Surface #########
num_surf_pts = 76

# a = 5
# b = 7
# e = rectangle(a,b)
# custom_dimensions = np.array([
#     [-4.,4.],
#     [-5.6,5.6]])

centers = [[-13.,-0.5],[-7.,2.],[2.,0.],[10.,-4.]]
radii = [2.,2.,4.,3.]
e = multi_circle(centers,radii)
custom_dimensions = np.array([
    [-18.,18.],
    [-9,6]])

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
    num_cps[i] = int(np.round(frac[i]*max_cps)+order-1)

######### Initialize Volume #########
Func = MyProblem(pts, normals, num_cps, order, custom_dimensions, exact=exact)
scaling, bases_surf, bases_curv = Func.get_values()

if visualize_init:
    Func.visualize_current()
    plt.show()
    exit()

# Key vector sizes
num_cps_pts = len(Func.cps)
num_hess_pts = len(Func.u['hess'])
num_surf_pts = len(Func.u['surf'])
print('Num_hess_pts: ', num_hess_pts)
print('Num_ctrl_pts: ', num_cps_pts)
print('Num_surf_pts: ', num_surf_pts,'\n')
#################################
EnergyMinModel = om.Group()
EnergyMinModel.add_subsystem('Curvature_Sampling', curv_sampling(
        num_cps=num_cps_pts,
        num_pts=num_hess_pts,
        dim=dim,
        scaling=scaling,
        bases=bases_curv
    ),promotes=['*'])
EnergyMinModel.add_subsystem('Surface_Sampling',surf_sampling(
        num_cps=num_cps_pts,
        num_pts=num_surf_pts,
        dim=dim,
        scaling=scaling,
        bases=bases_surf,
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
        bbox_diag=Func.Bbox_diag,
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
Prob.driver.opt_settings['Iterations limit'] = 150000
Prob.driver.opt_settings['Major optimality tolerance'] = tol
Prob.setup()
#################################
if prev_filename:
    previous_data = pickle.load(open(prev_filename,"rb"))
    Prob['phi_cps'] = previous_data.cps[:,2]/previous_data.Bbox_diag
else:
    Prob['phi_cps'] = Func.cps[:,2]
#################################
Prob.run_model()
t1 = time.time()
Prob.run_driver()
t2 = time.time()
#################################
print('Runtime: ',t2-t1)
print('Final Objective Value: ',Prob['objective'])
Func.runtime = t2-t1
Func.set_cps(Prob['phi_cps']*Func.Bbox_diag)
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