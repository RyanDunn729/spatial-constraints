from modules.objective import Objective
from modules.curv_samp import curv_sampling
from modules.surf_samp import surf_sampling
from skimage.measure import marching_cubes
from modules.fnorm import Fnorm
from modules.base import MyProblem
from stl.mesh import Mesh
import openmdao.api as om
import numpy as np
import time

class model(object):

    def __init__(self,max_cps,border,dim,tol,exact):
        self.max_cps = max_cps
        self.border = border
        self.dim = dim
        self.tol = tol
        self.exact = exact

    def inner_solve(self,surf_pts,normals,Lr,Ln,Lp,order,init_manual=None):
        print('Lr={}'.format(Lr))
        print('Ln={}'.format(Ln))
        print('Lp={}'.format(Lp))
        dim = self.dim
        ######### Initialize Volume #########
        Func = MyProblem(surf_pts, normals, self.max_cps, self.border, order, exact=self.exact, )
        scaling = Func.scaling
        phi_init = Func.cps[:,3]
        # Key vector sizes
        num_cps_pts  = Func.num_cps_pts
        num_hess_pts = Func.num_hess_pts
        num_surf_pts = Func.num_surf_pts
        #################################
        bases_surf = [
            Func.get_basis(loc='surf',du=0,dv=0,dw=0),
            Func.get_basis(loc='surf',du=1,dv=0,dw=0),
            Func.get_basis(loc='surf',du=0,dv=1,dw=0),
            Func.get_basis(loc='surf',du=0,dv=0,dw=1),
        ]
        bases_curv = [
            Func.get_basis(loc='hess',du=2,dv=0,dw=0),
            Func.get_basis(loc='hess',du=1,dv=1,dw=0),
            Func.get_basis(loc='hess',du=0,dv=2,dw=0),
            Func.get_basis(loc='hess',du=1,dv=0,dw=1),
            Func.get_basis(loc='hess',du=0,dv=1,dw=1),
            Func.get_basis(loc='hess',du=0,dv=0,dw=2),
        ]
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
        Prob.driver.opt_settings['Major optimality tolerance'] = self.tol
        Prob.setup()
        #################################
        if init_manual is not None:
            Prob['phi_cps'] = init_manual
        else:
            Prob['phi_cps'] = phi_init
        #################################
        Prob.run_model()
        t1 = time.time()
        Prob.run_driver()
        t2 = time.time()
        #################################
        print('Runtime: ',t2-t1)
        Func.runtime = t2-t1
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