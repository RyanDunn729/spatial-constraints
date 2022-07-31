from modules.soft_objective import soft_objective
from modules.surf_samp import surf_sampling
from modules.curv_samp import curv_sampling
from modules.assm_hess import assm_hess
from modules.base import MyProblem
import openmdao.api as om
import omtools.api as ot
import numpy as np
import time

class model(object):

    def __init__(self,e,max_cps,R,border,dim,soft_const,tol):
        self.e = e
        self.max_cps = max_cps
        self.R = R
        self.border = border
        self.dim = dim
        self.soft_const = soft_const
        self.tol = tol

    def inner_solve(self,num_surf_pts,L1,L2,L3,order,init_manual=None):
        e = self.e
        max_cps = self.max_cps
        R = self.R
        border = self.border
        dim = self.dim

        pts = e.points(num_surf_pts)
        normals = e.unit_pt_normals(num_surf_pts)

        num_exact = 10000
        ex_pts = e.points(num_exact)
        ex_norms = e.unit_pt_normals(num_exact)
        exact = np.stack((ex_pts,ex_norms))

        ######### Initialize Volume #########
        Func = MyProblem(exact, pts, normals, max_cps, R, border, order)
        scaling, dA, A, bases_surf, bases_curv = Func.get_values()

        # Key vector sizes
        num_cps_pts = len(Func.cps)
        num_hess_pts = len(Func.u['hess'])
        num_surf_pts = len(Func.u['surf'])
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
                self.register_output('Fnorm',ot.pnorm(H,axis=(1,2)))
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
        if not self.soft_const:
            objective.add_subsystem('Curvature',Curvature_Objective(),promotes=['*'])
        elif self.soft_const:
            comp = surf_sampling(num_cps=num_cps_pts,num_pts=num_surf_pts,dim=dim,
                                scaling=scaling,bases=bases_surf)
            objective.add_subsystem('Surface_Sampling',comp,promotes=['*'])
            objective.add_subsystem('Fnorms',Fnorm(),promotes=['*'])
            comp = soft_objective(num_samp=num_hess_pts,
                                num_surf=num_surf_pts,dim=dim,normals=normals)
            objective.add_subsystem('Penals',comp,promotes=['*'])
        #################################
        if not self.soft_const:
            constraint = ot.Group()
            comp = surf_sampling(num_cps=num_cps_pts,num_pts=num_surf_pts,dim=dim,
                                scaling=scaling,bases=bases_surf)
            constraint.add_subsystem('Surface_projection',comp,promotes=['*'])
        #################################
        Prob = om.Problem()
        model = Prob.model
        model.add_subsystem('Inputs_Group', inputs, promotes=['*'])
        model.add_subsystem('Objective_Group', objective, promotes=['*'])
        if not self.soft_const:
            model.add_subsystem('Constraints_Group', constraint, promotes=['*'])

        Prob.model.add_design_var('phi_cps',lower=-1,upper=1)
        if self.soft_const:
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
        Prob.driver.opt_settings['Iterations limit'] = 50000
        Prob.driver.opt_settings['Major feasibility tolerance'] = 1e-12
        Prob.driver.opt_settings['Major optimality tolerance'] = self.tol
        Prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-12
        #################################
        Prob.setup(force_alloc_complex=True)
        #################################
        if init_manual is not None:
            Prob['phi_cps'] = init_manual
        else:
            Prob['phi_cps'] = Func.cps[:,2]
        #################################
        Prob.run_model()
        t1 = time.time()
        Prob.run_driver()
        t2 = time.time()
        #################################
        print('Runtime: ',t2-t1)
        Func.runtime = t2-t1
        Func.E, Func.E_scaled = Func.get_energy_terms(Prob)
        Func.set_cps(Prob['phi_cps']*Func.Bbox_diag)
        if self.soft_const:
            print('Final Objective Value: ',Prob['soft_objective'])
        else:
            print('Final Objective Value: ',Prob['Curvature_Metric'])
        phi = Func.eval_surface()
        print('Surface error: ', np.sqrt(np.sum(phi**2)/len(Func.exact[0])))
        print('END')
        return Func

    def repeating_opt(self,num_surf_pts,L1,L2,L3,order):
        e = self.e
        max_cps = self.max_cps
        R = self.R
        border = self.border
        dim = self.dim

        pts = e.points(num_surf_pts)
        normals = e.unit_pt_normals(num_surf_pts)

        num_exact = 10000
        ex_pts = e.points(num_exact)
        ex_norms = e.unit_pt_normals(num_exact)
        exact = np.stack((ex_pts,ex_norms))

        ######### Initialize Volume #########
        Func = MyProblem(exact, pts, normals, max_cps, R, border, order)
        scaling, dA, A, bases_surf, bases_curv = Func.get_values()

        # Key vector sizes
        num_cps_pts = len(Func.cps)
        num_hess_pts = len(Func.u['hess'])
        num_surf_pts = len(Func.u['surf'])
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
                self.register_output('Fnorm',ot.pnorm(H,axis=(1,2)))
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
        if not self.soft_const:
            objective.add_subsystem('Curvature',Curvature_Objective(),promotes=['*'])
        elif self.soft_const:
            comp = surf_sampling(num_cps=num_cps_pts,num_pts=num_surf_pts,dim=dim,
                                scaling=scaling,bases=bases_surf)
            objective.add_subsystem('Surface_Sampling',comp,promotes=['*'])
            objective.add_subsystem('Fnorms',Fnorm(),promotes=['*'])
            comp = soft_objective(num_samp=num_hess_pts,
                                num_surf=num_surf_pts,dim=dim,normals=normals)
            objective.add_subsystem('Penals',comp,promotes=['*'])
        #################################
        if not self.soft_const:
            constraint = ot.Group()
            comp = surf_sampling(num_cps=num_cps_pts,num_pts=num_surf_pts,dim=dim,
                                scaling=scaling,bases=bases_surf)
            constraint.add_subsystem('Surface_projection',comp,promotes=['*'])
        #################################
        Prob = om.Problem()
        model = Prob.model
        model.add_subsystem('Inputs_Group', inputs, promotes=['*'])
        model.add_subsystem('Objective_Group', objective, promotes=['*'])
        if not self.soft_const:
            model.add_subsystem('Constraints_Group', constraint, promotes=['*'])

        Prob.model.add_design_var('phi_cps',lower=-1,upper=1)
        if self.soft_const:
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
        Prob.driver.opt_settings['Iterations limit'] = 50000
        Prob.driver.opt_settings['Major feasibility tolerance'] = 1e-12
        Prob.driver.opt_settings['Major optimality tolerance'] = self.tol
        Prob.driver.opt_settings['Minor feasibility tolerance'] = 1e-12
        #################################
        Prob.setup(force_alloc_complex=True)
        #################################
        Prob['phi_cps'] = Func.cps[:,2]
        #################################
        Prob.run_model()
        t1 = time.time()
        Prob.run_driver()
        t2 = time.time()
        #################################
        print('Runtime: ',t2-t1)
        Func.runtime = t2-t1
        Func.E, Func.E_scaled = Func.get_energy_terms(Prob)
        Func.set_cps(Prob['phi_cps']*Func.Bbox_diag)
        if self.soft_const:
            print('Final Objective Value: ',Prob['soft_objective'])
        else:
            print('Final Objective Value: ',Prob['Curvature_Metric'])
        phi = Func.eval_surface()
        print('Surface error: ', np.sqrt(np.sum(phi**2)/len(Func.exact[0])))
        print('END')
        del Prob
        return Func
