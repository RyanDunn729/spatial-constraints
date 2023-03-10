from lsdo_geo.bsplines.bspline_surface import BSplineSurface
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MyProblem(object):
    def __init__(self, surf_pts, normals, num_cps, order, dimensions, exact=None):
        k = 6 # nearest neighbors for initialization
        self.order = order
        self.exact = exact
        self.u = {}
        self.v = {}
        if (surf_pts[:,0] < dimensions[0,0]).any() \
            or (surf_pts[:,0] > dimensions[0,1]).any() \
            or (surf_pts[:,1] < dimensions[1,0]).any() \
            or (surf_pts[:,1] > dimensions[1,1]).any():
            raise ValueError("surface points lie outside of the defined dimensions")
            
        self.surf_pts = surf_pts
        self.normals = normals
        self.dimensions = dimensions
        self.num_cps = num_cps
        self.area = np.product(np.diff(dimensions))
        dxy = np.diff(dimensions).flatten()
        self.scaling = 1/dxy

        # Minimum Bounding Box Diagonal
        lower = np.min(surf_pts,axis=0)
        upper = np.max(surf_pts,axis=0)
        diff = upper-lower
        self.Bbox_diag = np.linalg.norm(diff)

        # Get initial control points
        self.cps = self.init_cps_Hicken(k=k,rho=10)

        # Standard uniform knot vectors
        kv_u = self.std_uniform_knot_vec(num_cps[0], order)
        kv_v = self.std_uniform_knot_vec(num_cps[1], order)
        # Define Bspline Volume object
        self.Surface = BSplineSurface('name',order,order,kv_u,kv_v,num_cps,self.cps)
        
        ### Get u,v for surface points ###
        self.u['surf'], self.v['surf'] = self.spatial_to_parametric(surf_pts)
        
        ### Get u,v quadrature points for evaluating Er (same as control points)
        temp_u, temp_v = self.spatial_to_parametric(self.cps[:,0:2])
        mask = np.argwhere(
            (temp_u>=0)*(temp_u<=1)*\
            (temp_v>=0)*(temp_v<=1)
        )
        self.u['hess'], self.v['hess'] = temp_u[mask].flatten(), temp_v[mask].flatten()

        ### Get Scaling Values ###
        num_hess = len(self.u['hess'])
        self.dA = self.area/(num_hess)

        self.num_surf_pts = int(len(surf_pts))
        self.num_cps_pts = int(np.product(num_cps))
        self.num_hess_pts = int(num_hess)

        print('BBox with border: \n',self.dimensions,'\n')
        print('BBox diagonal: ',self.Bbox_diag,'\n')
        print('Num_surf_pts: ', len(surf_pts))
        print('num_cps: ',num_cps,'=',np.product(num_cps))
        print('num_hess: ',num_hess,'=',np.product(num_hess),'\n')
        print('phi0_min: ',np.min(self.cps[:,2]))
        print('phi0_max: ',np.max(self.cps[:,2]),'\n')
        print('Order: ',order,'\n')

    def set_cps(self, cps_phi):
        self.cps[:,2] = cps_phi
        self.Surface.control_points = self.cps
        return

    def spatial_to_parametric(self,pts):
        param = np.empty((2,len(pts)))
        for i in range(2):
            param[i] = (pts[:,i] - self.dimensions[i,0]) / np.diff(self.dimensions[i,:])[0]
        return param[0], param[1]

    def eval_pts(self,pts,du=0,dv=0):
        u,v = self.spatial_to_parametric(pts)
        b = self.Surface.get_basis_matrix(u,v,0,0)
        return b.dot(self.cps[:,2])

    def gradient_eval(self,pts):
        dxy = np.diff(self.dimensions).flatten()
        scaling = 1/dxy
        u,v = self.spatial_to_parametric(pts)
        bdx = self.Surface.get_basis_matrix(u,v,1,0)
        dpdx = bdx.dot(self.cps[:,2])*scaling[0]
        bdy = self.Surface.get_basis_matrix(u,v,0,1)
        dpdy = bdy.dot(self.cps[:,2])*scaling[1]
        return dpdx, dpdy

    def std_uniform_knot_vec(self,num_cps,order):
        knot_vector = np.zeros(num_cps + order)
        for i in range(num_cps + order):
            knot_vector[i] = (i - order + 1) / (num_cps - order + 1)
        return knot_vector

    def init_cps_Hicken(self,k=6,rho=10):
        rangex = self.dimensions[0]
        rangey = self.dimensions[1]
        # Order 4, index 1.0: basis = [1/6, 4/6, 1/6]
        # Order 5, index 1.5: basis = [1/24, 11/24, 11/24, 1/24]
        # Order 6, index 2.0: basis = [1/120, 26/120, 66/120, 26/120, 1/120]
        Q = (self.order-2)/2
        A = np.array([[self.num_cps[0]-1-Q, Q],
                    [Q, self.num_cps[0]-1-Q]])
        b = np.array([rangex[0]*(self.num_cps[0]-1), rangex[1]*(self.num_cps[0]-1)])
        xn = np.linalg.solve(A,b)
        A = np.array([[self.num_cps[1]-1-Q, Q],
                    [Q, self.num_cps[1]-1-Q]])
        b = np.array([rangey[0]*(self.num_cps[1]-1), rangey[1]*(self.num_cps[1]-1)])
        yn = np.linalg.solve(A,b)

        cps = np.zeros((np.product(self.num_cps), 3))
        cps[:, 0] = np.einsum('i,j->ij', np.linspace(xn[0],xn[1],self.num_cps[0]), np.ones(self.num_cps[1])).flatten()
        cps[:, 1] = np.einsum('i,j->ij', np.ones(self.num_cps[0]), np.linspace(yn[0],yn[1],self.num_cps[1])).flatten()

        best_pts = self.surf_pts
        best_nrms = self.normals
        if self.exact is not None:
            best_pts = self.exact[0]
            best_nrms = self.exact[1]
        dataset = KDTree(best_pts)
        distances,indices = dataset.query(cps[:,0:2],k=k)
        d_norm = np.transpose(distances.T - distances[:,0]) + 1e-16
        exp = np.exp(-rho*d_norm)
        Dx = dataset.data[indices] - np.reshape(cps[:,0:2],(np.product(self.num_cps),1,2))
        phi = np.einsum('ijk,ij->i',Dx*best_nrms[indices],exp)/np.sum(exp,axis=1)
        np.random.seed(1)
        phi += 1e-3*self.Bbox_diag*(2*np.random.rand(np.product(self.num_cps))-1)
        cps[:,2] = phi/self.Bbox_diag
        return cps

    def get_basis(self,loc='surf',du=0,dv=0):
        basis = self.Surface.get_basis_matrix(self.u[loc],self.v[loc],du,dv)
        return basis

    def eval_surface(self):
        sample_pts = self.surf_pts
        if self.exact is not None:
            sample_pts = self.exact[0]
        u,v = self.spatial_to_parametric(sample_pts)
        basis = self.Surface.get_basis_matrix(u,v,0,0)
        phi = basis.dot(self.cps[:,2])
        return phi

    def check_local_RMS_error(self,bbox_perc,res):
        ep_max = bbox_perc*self.Bbox_diag / 100
        ep_range = np.linspace(-ep_max,ep_max,res)
        sample_pts = self.surf_pts
        sample_normals = self.normals
        if self.exact is not None:
            sample_pts = self.exact[0]
            sample_normals = self.exact[1]
        dataset = KDTree(sample_pts)
        RMS_local = np.zeros(len(ep_range))
        for i,ep in enumerate(ep_range):
            i_pts = sample_pts + ep*sample_normals
            u,v = self.spatial_to_parametric(i_pts)
            b = self.Surface.get_basis_matrix(u,v,0,0)
            phi = b.dot(self.cps[:,2])
            phi_ex,_ = dataset.query(i_pts,k=1)
            RMS_local[i] = np.sqrt(np.sum(  (abs(phi)-phi_ex)**2  )/len(phi_ex))
        return np.linspace(-bbox_perc,bbox_perc,res), RMS_local/self.Bbox_diag