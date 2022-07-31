from modules.soft_objective import soft_objective
from modules.multi_circle import multi_circle
from modules.surf_samp import surf_sampling
from modules.curv_samp import curv_sampling
from modules.assm_hess import assm_hess
from modules.rectangle import rectangle
from modules.ellipse import ellipse
from modules.base import MyProblem
from lsdo_viz.api import Problem
import matplotlib.pyplot as plt
import openmdao.api as om
import omtools.api as ot
import numpy as np
import pickle
import time
print('Imported Packages \n')

######### Configurables #########
dim = 2
order = 4
R = 1.5
max_cps = 96
border = 0.25

soft_const = True
L1 = 1.
L2 = 100.
L3 = 1.

tol = 1e-4

prev_filename = None
prev_filename = "SAVED_DATA/Opt_rectangle_L2_1.pkl"
# prev_filename = "_Saved_Function.pkl"

visualize_init = False
######### Get Contour #########
num_surf_pts = 100
a = 5
b = 7
centers = [[-13.,-0.5],[-7.,2.],[2.,0.],[10.,-4.]]
radii = [2.,2.,4.,3.]

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

######### Initialize Volume #########
Func = MyProblem(exact, pts, normals, max_cps, R, border, order)
scaling, dA, A, bases_surf, bases_curv = Func.get_values()
Func.a = a
Func.b = b
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
class Curvature_Objective(ot.Group):
    def setup(self):
        H = self.declare_input('hessians',shape=(num_hess_pts,dim,dim))
        dA = self.declare_input('dA',shape=(1,))
        A = self.declare_input('A',shape=(1,))
        Fnorm = ot.pnorm(H,axis=(1,2))
        self.register_output('Curvature_Metric',ot.sum(Fnorm**2)*dA/A)
class Fnorm(ot.Group):
    def setup(self):
        H = self.declare_input('hessians',shape=(num_hess_pts,dim,dim))
        self.register_output('Fnorm',ot.pnorm(H,axis=(1,2),pnorm_type=2))
#################################
inputs = ot.Group()
inputs.create_indep_var('phi_cps', shape=(num_cps_pts,))
inputs.create_indep_var('dA',val=dA)
inputs.create_indep_var('A',val=A)
inputs.create_indep_var('lambdas',val=np.array([L1,L2,L3]))
#################################
objective = ot.Group()
objective.add_subsystem('Curvature_Sampling', curv_sampling(num_cps=num_cps_pts,
                        num_pts=num_hess_pts,dim=dim,scaling=scaling,bases=bases_curv),
                        promotes=['*'])
objective.add_subsystem('Assemble_Hessians',assm_hess(num_pts=num_hess_pts,dim=dim),
                        promotes=['*'])
if not soft_const:
    objective.add_subsystem('Curvature',Curvature_Objective(),promotes=['*'])
elif soft_const:
    comp = surf_sampling(num_cps=num_cps_pts,num_pts=num_surf_pts,dim=dim,
                        scaling=scaling,bases=bases_surf)
    objective.add_subsystem('Surface_Sampling',comp,promotes=['*'])
    objective.add_subsystem('Fnorms',Fnorm(),promotes=['*'])
    comp = soft_objective(num_samp=num_hess_pts,
                          num_surf=num_surf_pts,dim=dim,normals=normals)
    objective.add_subsystem('Penals',comp,promotes=['*'])
#################################
if not soft_const:
    constraint = ot.Group()
    comp = surf_sampling(num_cps=num_cps_pts,num_pts=num_surf_pts,dim=dim,
                         scaling=scaling,bases=bases_surf)
    constraint.add_subsystem('Surface_projection',comp,promotes=['*'])
#################################
Prob = Problem()
model = Prob.model
model.add_subsystem('Inputs_Group', inputs, promotes=['*'])
model.add_subsystem('Objective_Group', objective, promotes=['*'])
if not soft_const:
    model.add_subsystem('Constraints_Group', constraint, promotes=['*'])

Prob.model.add_design_var('phi_cps',lower=-1,upper=1)
if soft_const:
    Prob.model.add_objective('soft_objective',scaler=1)
else:
    Prob.model.add_objective('Curvature_Metric',scaler=1)
    Prob.model.add_constraint('phi_surf',equals=np.zeros(num_surf_pts),linear=True)
    Prob.model.add_constraint('dpdx_surf',equals=-normals[:,0],linear=True)
    Prob.model.add_constraint('dpdy_surf',equals=-normals[:,1],linear=True)
#################################
Prob.driver = om.pyOptSparseDriver()
Prob.driver.options['optimizer'] = 'SNOPT'
Prob.driver.opt_settings['Major iterations limit'] = 10000
Prob.driver.opt_settings['Minor iterations limit'] = 10000
Prob.driver.opt_settings['Iterations limit'] = 150000
Prob.driver.opt_settings['Major feasibility tolerance'] = 1e-12
Prob.driver.opt_settings['Major optimality tolerance'] = tol
Prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-12
#################################
Prob.setup(force_alloc_complex=True)
#################################
if prev_filename:
    previous_data = pickle.load(open(prev_filename,"rb"))
    Prob['phi_cps'] = previous_data.cps[:,2]/previous_data.Bbox_diag
else:
    Prob['phi_cps'] = Func.cps[:,2]
#################################
t1 = time.time()
Prob.run()
t2 = time.time()
#################################
print('Runtime: ',t2-t1)
if soft_const:
    print('Final Objective Value: ',Prob['soft_objective'])
else:
    print('Final Objective Value: ',Prob['Curvature_Metric'])
Func.runtime = t2-t1
Func.E, Func.E_scaled = Func.get_energy_terms(Prob)
Func.set_cps(Prob['phi_cps']*Func.Bbox_diag)
pickle.dump(Func, open( "_Saved_Function.pkl","wb"))
phi = Func.eval_surface()
print('Surface error: ', np.sqrt(np.sum(phi**2)/len(Func.exact[0])))
print('END')