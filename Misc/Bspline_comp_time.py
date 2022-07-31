import pickle
import openmdao.api as om
import omtools.api as ot
import numpy as np
from time import time
import matplotlib.pyplot as plt
import seaborn as sns

class BsplineVolume(om.ExplicitComponent):

    def initialize(self):
        # Additional derivatives to define 1st order conversion to 
        self.options.declare('dudx', default=1., types=float)
        self.options.declare('dvdy', default=1., types=float)
        self.options.declare('dwdz', default=1., types=float)
        # Number of points to be sampled (for 1 config)
        self.options.declare('num_pt', default=100, types=int)
        # Number of configurations total
        self.options.declare('k',default=3, types=int)

    def setup(self):
        k = self.options['k']
        num_pt = self.options['num_pt']
        # Input points
        self.add_input('pt',shape=(num_pt,k,3))
        # Signedfun Output: 
        # (+) When on the interior
        # (-) When on the exterior
        # (0) When on the surface of anatomy
        self.add_output('signedfun',shape=(num_pt,k,))

    def setup_partials(self):
        k = self.options['k']
        num_pt = self.options['num_pt']
        # Col_ind = 0,1,2,3,4 ...
        col_ind = np.arange(0,3*k*num_pt)
        # Row_ind = 0,0,0, 1,1,1, 2,2,2 ...
        row_ind = np.empty(num_pt*k*3)
        for i in range(num_pt*k):
            row_ind[3*i:3*(i+1)] = i*np.ones(3)
        # Sparse Jacobian coordinates (Diagonal)
        self.declare_partials(of='signedfun', wrt='pt', rows=row_ind, cols=col_ind)

    def compute(self, inputs, outputs):
        # Compute signedfun
        k = self.options['k']
        num_pt = self.options['num_pt']
        # Project physical coordinates to parametric coordinates
        # (x,y,z) -> (u,v,w) 
        u,v,w = Func.Volume.project(inputs['pt'].reshape(num_pt*k,3),350,0)
        # Get Bspline Matrix (sparse)
        # (u,v,w) -> (x,y,z,phi)
        basis = Func.Volume.get_basis_matrix(u,v,w,0,0,0)
        # Only dot 4th col of ctrl_pts bc we only want (phi)
        # (u,v,w) -> (_,_,_,phi)
        outputs['signedfun'] = basis.dot(Func.cps[:,3]).reshape(num_pt,k)

    def compute_partials(self, inputs, partials):
        # Compute the partial derivatives
        k = self.options['k']
        num_pt = self.options['num_pt']
        u,v,w = Func.Volume.project(inputs['pt'].reshape(num_pt*k,3),350,0)
        # Get basis matrices for du, dv, dw (1st derivatives)
        basis100 = Func.Volume.get_basis_matrix(u,v,w,1,0,0)
        basis010 = Func.Volume.get_basis_matrix(u,v,w,0,1,0)
        basis001 = Func.Volume.get_basis_matrix(u,v,w,0,0,1)
        # By chain rule: 
        # dphi_dx = dphi_du * du_dx
        # dphi_dy = dphi_dv * dv_dy
        # dphi_dz = dphi_dw * dw_dz
        dp_dx = basis100.dot(Func.cps[:,3]) * self.options['dudx']
        dp_dy = basis010.dot(Func.cps[:,3]) * self.options['dvdy']
        dp_dz = basis001.dot(Func.cps[:,3]) * self.options['dwdz']
        # Return vector of values going dx,dy,dz, dx,dy,dz, ... etc
        partials['signedfun','pt'] = np.column_stack((dp_dx,dp_dy,dp_dz)).flatten()

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    import pickle
    import numpy as np
    from skimage.measure import marching_cubes

    Func = pickle.load( open( "_Saved_Function.p", "rb" ) )
    phi_cps = pickle.load( open( "_Saved_phi_cps.p", "rb" ) )
    Func.set_cps(phi_cps)
    dudx = Func.inv_scaling_matrix[0,0]
    dvdy = Func.inv_scaling_matrix[1,1]
    dwdz = Func.inv_scaling_matrix[2,2]

    dat = [3,6,14,26,50,100]
    evaltime = np.empty(len(dat))
    num_pts = np.empty(len(dat))
    for j,res in enumerate(dat):
        ### Sample a (res)^3 grid across the bounding volume avoiding the barriers
        u_total = np.einsum('i,j,k->ijk', np.linspace(0, 1, res+2)[1:res+1], np.ones(res),np.ones(res)).flatten()
        v_total = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(0, 1, res+2)[1:res+1],np.ones(res)).flatten()
        w_total = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(0, 1, res+2)[1:res+1]).flatten()
        basis = Func.Volume.get_basis_matrix(u_total, v_total, w_total, 0, 0, 0)
        verts = basis.dot(Func.cps[:,0:3])

        ### Get num_pt and define "k" configurations ##
        num_pts[j] = res**3
        num_pt = res**3
        k = 1
        print('num_pt: ',num_pt)
        # print('k: ',k)
        pts = np.empty((num_pt,k,3))
        for i in range(k):
            pts[:,i,:] = verts

        group = Group()    
        comp = BsplineVolume(dudx=dudx,dvdy=dvdy,dwdz=dwdz,num_pt=num_pt,k=k)
        group.add_subsystem('BsplineVolume', comp, promotes = ['*'])
            
        prob = Problem()
        prob.model = group

        prob.setup()

        prob['pt'] = pts
        t1 = time()
        prob.run_model()
        t2 = time()
        evaltime[j] = t2-t1

    sns.set()

    plt.loglog(num_pts,evaltime,'.-')
    plt.xlabel('Number of Points Evaluated')
    plt.ylabel('Evaluation Time (s)')
    plt.show()

    # prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    # prob.check_partials(compact_print=True)

### For implementation ###
# from Volume_comp import BsplineVolume

# comp = BsplineVolume(dudx=dudx,dvdy=dvdy,dwdz=dwdz,num_pt=num_pt,k=k)
# Prob.model.add_subsystem('sdf_Bspline',comp,promotes=['*'])
# Prob.model.add_constraint('signedfun',lower=0,linear=True)