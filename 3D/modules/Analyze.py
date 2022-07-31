from modules.soft_objective import soft_objective
from modules.curv_samp import curv_sampling
from modules.surf_samp import surf_sampling
from skimage.measure import marching_cubes
from modules.assm_hess import assm_hess
from modules.fnorm import Fnorm
from modules.base import MyProblem
from stl.mesh import Mesh
import openmdao.api as om
import omtools.api as ot
import numpy as np
import time

class model(object):

    def __init__(self,max_cps,R,border,dim,tol,exact,soft_const):
        self.max_cps = max_cps
        self.R = R
        self.border = border
        self.dim = dim
        self.tol = tol
        self.exact = exact
        self.soft_const = soft_const

    def inner_solve(self,surf_pts,normals,L1,L2,L3,order,init_manual=None):
        print('L1={}'.format(L1))
        print('L2={}'.format(L2))
        print('L3={}'.format(L3))
        dim = self.dim
        soft_const = self.soft_const
        ######### Initialize Volume #########
        Func = MyProblem(self.exact, surf_pts, normals, self.max_cps, self.R, self.border, order)
        scaling, dV, V, bases_surf, bases_curv = Func.get_values()

        # Key vector sizes
        num_cps_pts = Func.num_cps_pts
        num_hess_pts = Func.num_hess_pts
        num_surf_pts = len(surf_pts)

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
        Prob.driver.opt_settings['Iterations limit'] = 50000
        Prob.driver.opt_settings['Major feasibility tolerance'] = 1e-9
        Prob.driver.opt_settings['Major optimality tolerance'] = self.tol
        Prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-9
        #################################
        Prob.setup(force_alloc_complex=True)
        #################################
        if init_manual:
            Prob['phi_cps'] = init_manual
        else:
            Prob['phi_cps'] = Func.cps[:,3]
        #################################
        Prob.run_model()
        t1 = time.time()
        Prob.run_driver()
        t2 = time.time()
        #################################
        print('Runtime: ',t2-t1)
        Func.runtime = t2-t1
        Func.E, Func.E_norm = Func.get_energy_terms(Prob)
        print('Energies: ',Func.E)
        print('Scaled Energies: ',Func.E_norm)
        Func.set_cps(Prob['phi_cps']*Func.Bbox_diag)
        del Prob
        return Func

    def generate_stl(self,Func,save_filename):
        res = 200
        u = np.einsum('i,j,k->ijk', np.linspace(0,1,res), np.ones(res),np.ones(res)).flatten()
        v = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(0,1,res),np.ones(res)).flatten()
        w = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(0,1,res)).flatten()
        basis = Func.Volume.get_basis_matrix(u, v, w, 0, 0, 0)
        phi = basis.dot(Func.cps[:,3]).reshape((res,res,res))
        verts, faces,_,_ = marching_cubes(phi, 0)
        verts = verts*np.diff(Func.dimensions).flatten()/(res-1) + Func.dimensions[:,0]
        surf = Mesh(np.zeros(faces.shape[0], dtype=Mesh.dtype))
        for i, f in enumerate(faces):
                for j in range(3):
                        surf.vectors[i][j] = verts[f[j],:]
        surf.save(save_filename)
        del verts,faces,surf,phi,u,v,w