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
L2 = 1e1  # Normal weighting
L3 = 1e3 # Surface weighting

### Fuselage ###
# max_cps = 47
# flag = 'Fuselage'
# tol = 1e-5

### Human ###
# max_cps = 42
# flag = 'Human'
# tol = 1e-5

### BUNNY ###
# max_cps = 42
# flag = 'Bunny'
# tol = 1e-5

### HEART ###
# max_cps = 47
# flag = 'Heart'
# tol = 1e-5

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

### Armadillo ###
# max_cps = 31
# flag = 'Armadillo'
# tol = 1e-4

### Buddha ###
# max_cps = 37
# flag = 'Buddha'
# tol = 1e-4

### BUNNY ###
max_cps = 29
flag = 'Bunny'
tol = 1e-5

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
filename = 'stl-files/Bunny_63802.stl'
# filename = 'stl-files/Bunny_100002.stl'

# filename = 'stl-files/Heart_5002.stl'

# filename = 'stl-files/Fuselage_5002.stl'

# filename = 'stl-files/Human_5002.stl'

# filename = 'stl-files/Battery.stl'
# filename = 'stl-files/Luggage_reduced.stl'

# filename = 'stl-files/wing.stl'

# filename = 'stl-files/dragon_100k.stl'
# filename = 'stl-files/buddha_5794.stl'
# filename = 'stl-files/armadillo_6002.stl'

# max_cps = 40
# flag = 'Bunny'
# tol = 1e-4

######### Get Surface #########
if flag == 'Bunny':
    if filename == 'stl-files/Bunny_100002.stl':
        data = pickle.load( open( "SAVED_DATA/_Bunny_data_100002.pkl", "rb" ) )
        surf_pts = data[0]
        normals = data[1]
    elif filename == 'stl-files/Bunny_63802.stl':
        data = pickle.load( open( "SAVED_DATA/_Bunny_data_63802.pkl", "rb" ) )
        surf_pts = data[0]
        normals = data[1]
    else:
        surf_pts, normals = extract_stl_info(filename)
    exact = pickle.load( open( "SAVED_DATA/_Bunny_data_exact_.pkl", "rb" ) )
elif flag == 'Heart':
    surf_pts, normals = extract_stl_info(filename)
    exact = pickle.load( open( "SAVED_DATA/_Heart_data_exact_.pkl", "rb" ) )
elif flag == 'Ellipsoid':
    e = Ellipsoid(a,b,c)
    exact = np.stack((e.points(10000),e.unit_pt_normals(10000)))
    surf_pts = e.points(num_pts)
    normals = e.unit_pt_normals(num_pts)
elif flag == 'Fuselage':
    surf_pts, normals = extract_stl_info(filename)
    exact = pickle.load( open( "SAVED_DATA/_Fuselage_data_exact_.pkl", "rb" ) )
elif flag == 'Human':
    surf_pts, normals = extract_stl_info(filename)
    exact = pickle.load( open( "SAVED_DATA/_Human_data_exact_.pkl", "rb" ) )
elif flag == 'Custom':
    surf_pts, normals = extract_stl_info(filename)
    exact = np.stack((surf_pts,normals))
elif flag == 'Dragon':
    data = pickle.load( open( "SAVED_DATA/dragon_data_100k.pkl", "rb" ) )
    surf_pts = data[0]
    normals = data[1]
    exact = pickle.load( open( "SAVED_DATA/dragon_data_exact.pkl", "rb" ) )
elif flag == 'Armadillo':
    data = pickle.load( open( "SAVED_DATA/armadillo_data_100k.pkl", "rb" ) )
    surf_pts = data[0]
    normals = data[1]
    exact = pickle.load( open( "SAVED_DATA/armadillo_data_exact.pkl", "rb" ) )
elif flag == 'Buddha':
    data = pickle.load( open( "SAVED_DATA/buddha_data_100k.pkl", "rb" ) )
    surf_pts = data[0]
    normals = data[1]
    exact = pickle.load( open( "SAVED_DATA/buddha_data_exact.pkl", "rb" ) )
else:
    surf_pts, normals = extract_stl_info(filename)
    exact = pickle.load( open( "SAVED_DATA/{}_data_exact_.pkl".format(flag), "rb" ) )

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
# Prob = om.Problem()
Prob = Problem()
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
t1 = time.time()
Prob.run()
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
if flag == 'Dragon':
    pickle.dump(Func, open( "SAVED_DATA/Opt_dragon_.pkl","wb"))
elif flag == 'Armadillo':
    pickle.dump(Func, open( "SAVED_DATA/Opt_armadillo_.pkl","wb"))
elif flag == 'Buddha':
    pickle.dump(Func, open( "SAVED_DATA/Opt_buddha_.pkl","wb"))
elif flag == 'Bunny':
    pickle.dump(Func, open( "SAVED_DATA/Opt_bunny_.pkl","wb"))
pickle.dump(Func, open( "_Saved_Function.pkl","wb"))
phi = Func.eval_surface()
phi = phi/Func.Bbox_diag
# print(num_cps_pts)
# print(num_surf_pts)
print('Surface error: \n',
        'Max: ',np.max(phi),'\n',
        'RMS: ',np.sqrt(np.sum(phi**2)/len(phi)))
# ep_range,local_err = Func.check_local_RMS_error(2,10)
# print('local_RMS_error: \n',np.transpose(np.stack((ep_range,local_err),axis=0)))
print("Lambdas: ",L1,L2,L3)
print("num_pts: ",num_surf_pts)
print("num_cps: ",num_cps_pts)
print("flag: ",flag)
print('END')