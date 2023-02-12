from modules.soft_objective import soft_objective
from modules.read_stl import extract_stl_info
from modules.curv_samp import curv_sampling
from modules.surf_samp import surf_sampling
from modules.assm_hess import assm_hess
from modules.ellipsoid import Ellipsoid
from modules.base import MyProblem
from lsdo_viz.api import Problem
from modules.fnorm import Fnorm
import matplotlib.pyplot as plt
import openmdao.api as om
import omtools.api as ot
import numpy as np
import pickle
import time

print('Imported Packages \n')

######### Configurables #########
dim = 3
R = 1
order = 4
border = 0.15

soft_const = True
L1 = 5e-1 # Curvature weighting
L2 = 1e2 # Normal weighting
L3 = 1e3 # Surface weighting

### Fuselage ###
max_cps = 44
flag = 'Fuselage'
tol = 1e-4
filename = 'stl-files/Fuselage_25k.stl'

### Human ###
max_cps = 34
flag = 'Human'
tol = 5e-4
filename = 'stl-files/Human_25k.stl'

### Battery ###
max_cps = 44
flag = 'Battery'
tol = 5e-4
filename = 'stl-files/Battery_25k.stl'

### Luggage ###
max_cps = 41
flag = 'Luggage'
tol = 1e-4
filename = 'stl-files/Luggage_25k.stl'

### Wing ###
max_cps = 44
flag = 'Wing'
tol = 5e-4
filename = 'stl-files/Wing_25k.stl'

### BUNNY ###
max_cps = 28
flag = 'Bunny'
tol = 1e-5
filename = 'stl-files/Bunny_100000.stl'

### HEART Study ###
# max_cps = 34
# flag = 'Heart'
# tol = 1e-4
# filename = "stl-files/Heart_100k.stl"

### Ellipsoid ###
# max_cps = 42
# flag = 'Ellipsoid'
# tol = 1e-5
# num_pts = 500
# a = 8
# b = 7.25
# c = 5.75

### Dragon ###
# max_cps = 35
# flag = 'Dragon'
# tol = 1e-4
# filename = 'stl-files/dragon_100k.stl'

### Armadillo ###
# max_cps = 31
# flag = 'Armadillo'
# tol = 1e-4
# filename = 'stl-files/armadillo_100k.stl'

### Buddha ###
# max_cps = 37
# flag = 'Buddha'
# tol = 1e-4
# filename = 'stl-files/buddha_100k.stl'

## HEART For Optimization ###
# max_cps = 34
# flag = 'Heart'
# tol = 1e-4
# filename = "stl-files/heart_case03_final1.stl"

visualize_init = False

# filename_exact = 'stl-files/Bunny.stl'     # 129951 vertices
# filename = 'stl-files/Bunny_77.stl'
# filename = 'stl-files/Bunny_108.stl'
# filename = 'stl-files/Bunny_252.stl'
# filename = 'stl-files/Bunny_297.stl'
# filename = 'stl-files/Bunny_327.stl'
# filename = 'stl-files/Bunny_377.stl'
# filename = 'stl-files/Bunny_412.stl'
# filename = 'stl-files/Bunny_502.stl'
# filename = 'stl-files/Bunny_677.stl'
# filename = 'stl-files/Bunny_1002.stl'
# filename = 'stl-files/Bunny_5002.stl'
# filename = 'stl-files/Bunny_10002.stl'
# filename = 'stl-files/Bunny_25002.stl'
# filename = 'stl-files/Bunny_40802.stl'
# filename = 'stl-files/Bunny_63802.stl'
# filename = 'stl-files/Bunny_100002.stl'

######### Get Surface #########
if flag == 'Ellipsoid':
    e = Ellipsoid(a,b,c)
    exact = np.stack((e.points(10000),e.unit_pt_normals(10000)))
    surf_pts = e.points(num_pts)
    normals = e.unit_pt_normals(num_pts)
else:
    surf_pts, normals = extract_stl_info(filename)
    exact = extract_stl_info("stl-files/{}_exact.stl".format(flag))

######### Initialize Volume #########
Func = MyProblem(exact, surf_pts, normals, max_cps, R, border, order)
scaling, dV, V, bases_surf, bases_curv = Func.get_values()

# Key vector sizes
num_cps_pts = Func.num_cps_pts
num_hess_pts = Func.num_hess_pts
num_surf_pts = len(surf_pts)

if visualize_init:
    Func.visualize_current(30)
    plt.show()
    exit()

#################################
class Curvature_Objective(ot.Group):
    def setup(self):
        H = self.declare_input('hessians',shape=(num_hess_pts,dim,dim))
        dV = self.declare_input('dV',shape=(1,))
        V = self.declare_input('V',shape=(1,))
        Fnorm = ot.pnorm(H,axis=(1,2))
        self.register_output('Curvature_Metric',ot.sum(Fnorm**2)*dV/V)
#################################
inputs = ot.Group()
inputs.create_indep_var('phi_cps', shape=(num_cps_pts,))
inputs.create_indep_var('dV',val=dV)
inputs.create_indep_var('V',val=V)
inputs.create_indep_var('lambdas',val=np.array([L1,L2,L3]))
#################################
objective = ot.Group()
comp = curv_sampling(dim=dim,num_cps=num_cps_pts,num_pts=num_hess_pts,
                     bases=bases_curv,scaling=scaling)
objective.add_subsystem('Curvature_Samples',comp,promotes=['*'])
comp = assm_hess(dim=dim,num_pts=num_hess_pts)
objective.add_subsystem('Assemble_Hessians',comp,promotes=['*'])
if not soft_const:
    comp = Curvature_Objective()
    objective.add_subsystem('Curvature',comp,promotes=['*'])
elif soft_const:
    comp = surf_sampling(num_cps=num_cps_pts,num_pts=num_surf_pts,dim=dim,
                        scaling=scaling,bases=bases_surf)
    objective.add_subsystem('Surface_Sampling',comp,promotes=['*'])
    objective.add_subsystem('Fnorms',Fnorm(num_pts=num_hess_pts,dim=dim),promotes=['*'])
    comp = soft_objective(num_samp=num_hess_pts,num_surf=num_surf_pts,dim=dim,normals=normals)
    objective.add_subsystem('Penals',comp,promotes=['*'])
#################################
if not soft_const:
    constraint = ot.Group()
    comp = surf_sampling(num_cps=num_cps_pts,num_pts=num_surf_pts,dim=dim,
                        scaling=scaling,bases=bases_surf)
    constraint.add_subsystem('Surface_Sampling',comp,promotes=['*'])
#################################
Prob = om.Problem()
# Prob = Problem()
model = Prob.model
model.add_subsystem('Inputs_Group', inputs, promotes=['*'])
model.add_subsystem('Objective_Group', objective, promotes=['*'])
if not soft_const:
    model.add_subsystem('Constraints_Group', constraint, promotes=['*'])

model.add_design_var('phi_cps',lower=-1,upper=1)
if soft_const:
    model.add_objective('soft_objective',scaler=1)
else:
    model.add_objective('Curvature_Metric',scaler=1)
    model.add_constraint('phi_surf',equals=np.zeros(num_surf_pts),linear=True)
    model.add_constraint('dpdx_surf',equals=-normals[:,0],linear=True)
    model.add_constraint('dpdy_surf',equals=-normals[:,1],linear=True)
    model.add_constraint('dpdz_surf',equals=-normals[:,2],linear=True)
#################################
Prob.driver = om.pyOptSparseDriver()
Prob.driver.options['optimizer'] = 'SNOPT'
Prob.driver.opt_settings['Major iterations limit'] = 10000
Prob.driver.opt_settings['Minor iterations limit'] = 10000
# Prob.driver.opt_settings['Verify level'] = 0
Prob.driver.opt_settings['Iterations limit'] = 150000
Prob.driver.opt_settings['Major feasibility tolerance'] = 1e-12
Prob.driver.opt_settings['Major optimality tolerance'] = tol
Prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-12
#################################
Prob.setup(force_alloc_complex=True)
#################################
Prob['phi_cps'] = Func.cps[:,3]
#################################
Prob.run_model()
t1 = time.time()
Prob.run_driver()
# Prob.run()
t2 = time.time()
#################################
print('Runtime: ',t2-t1)
Func.runtime = t2-t1
if soft_const:
    print('Final Objective Value: ',Prob['soft_objective'])
else:
    print('Final Objective Value: ',Prob['Curvature_Metric'])
    print('Constraint check: \nmax_phi_surf: ',np.max(abs(Prob['phi_surf'])))
Func.set_cps(Prob['phi_cps']*Func.Bbox_diag)
Func.E, Func.E_scaled = Func.get_energy_terms(Prob)
print('Energies: ',Func.E)
print('Scaled Energies: ',Func.E_scaled)
# pickle.dump(Func, open( "SAVED_DATA/Opt_{}_.pkl".format(flag),"wb"))
# pickle.dump(Func, open( "_Saved_Function.pkl","wb"))
pickle.dump(Func, open( "Opt_{}_For_OffSurface3.pkl".format(flag),"wb"))
phi = Func.eval_surface()
phi = phi/Func.Bbox_diag
# print(num_cps_pts)
# print(num_surf_pts)
print('Surface error (rel): \n',
        'Max: ',np.max(phi),'\n',
        'RMS: ',np.sqrt(np.sum(phi**2)/len(phi)))
print('Surface error (units): \n',
        'Max: ',Func.Bbox_diag*np.max(phi),'\n',
        'RMS: ',Func.Bbox_diag*np.sqrt(np.mean(phi**2)))
# ep_range,local_err = Func.check_local_RMS_error(2,10)
# print('local_RMS_error: \n',np.transpose(np.stack((ep_range,local_err),axis=0)))
print("Lambdas: ",L1,L2,L3)
print("num_pts: ",num_surf_pts)
print("num_cps: ",num_cps_pts)
print("flag: ",flag)
print('END')