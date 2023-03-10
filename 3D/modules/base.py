from lsdo_geo.bsplines.bspline_volume import BSplineVolume
# from modules.Bspline_Volume import BSplineVolume
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

class MyProblem(object):

    def __init__(self, surf_pts, normals, max_cps, border, order, exact=None):
        self.order = order # Must be >2
        self.u = {}
        self.v = {}
        self.w = {}
        if exact is not None:
            self.exact = exact
        else:
            self.exact = (surf_pts,normals)
        self.surf_pts = surf_pts
        self.normals = normals

        # Minimum Bounding Box Diagonal
        lower = np.min(surf_pts,axis=0)
        upper = np.max(surf_pts,axis=0)
        diff = upper-lower
        self.Bbox_diag = np.linalg.norm(diff)
        self.dimensions = np.stack((lower-diff*border, upper+diff*border),axis=1)
        self.total_volume = np.product(np.diff(self.dimensions))
        dxyz = np.diff(self.dimensions).flatten()
        self.scaling = 1/dxyz

        # Scale the resolutions of cps and hess samples
        frac = dxyz / np.max(dxyz)
        num_cps = np.zeros(3,dtype=int)
        for i,ratio in enumerate(frac):
            if ratio < 0.75:
                ratio = 0.75
            num_cps[i] = int(np.round(frac[i]*max_cps)+order-1)
        self.num_cps = num_cps

        # Get initial control points
        self.cps = self.init_cps_Hicken(k=10,rho=10)

        # Standard uniform knot vectors
        kv_u = self.std_uniform_knot_vec(num_cps[0], order)
        kv_v = self.std_uniform_knot_vec(num_cps[1], order)
        kv_w = self.std_uniform_knot_vec(num_cps[2], order)
        # Define Bspline Volume object
        self.Volume = BSplineVolume('name',order,order,order,kv_u,kv_v,kv_w,num_cps,self.cps)

        # Get uvw_mesh
        self.u['surf'], self.v['surf'], self.w['surf'] = self.spatial_to_parametric(surf_pts)
        # Get uvw_hess
        temp_u, temp_v, temp_w = self.spatial_to_parametric(self.cps[:,0:3])
        mask = np.argwhere(
            (temp_u>=0)*(temp_u<=1)*\
            (temp_v>=0)*(temp_v<=1)*\
            (temp_w>=0)*(temp_w<=1)
        )
        self.u['hess'], self.v['hess'], self.w['hess'] = temp_u[mask].flatten(), temp_v[mask].flatten(), temp_w[mask].flatten()

        # Sizing
        num_hess = len(self.u['hess'])
        self.dV = self.total_volume/(num_hess)

        self.num_surf_pts = int(len(surf_pts))
        self.num_cps_pts  = int(np.product(num_cps))
        self.num_hess_pts = int(num_hess)

        print('BBox with border: \n',self.dimensions,'\n')
        print('BBox diagonal: ',self.Bbox_diag,'\n')
        print('Num_surf_pts: ', len(surf_pts))
        print('num_cps: ',num_cps,'=',self.num_cps_pts)
        print('num_hess: ',num_hess,'=',self.num_hess_pts,'\n')
        print('phi0_min: ',np.min(self.cps[:,3]))
        print('phi0_max: ',np.max(self.cps[:,3]),'\n')
        print('Order: ',order,'\n')

    def get_values(self):
        surf, curv = self.get_bases()
        return self.scaling, surf, curv

    def set_cps(self, cps_phi):
        self.cps[:,3] = cps_phi
        self.Volume.control_points = self.cps
        return

    def spatial_to_parametric(self,pts):
        param = np.empty((3,len(pts)))
        for i in range(3):
            param[i] = (pts[:,i] - self.dimensions[i,0]) / np.diff(self.dimensions[i,:])[0]
        return param[0], param[1], param[2]

    def std_uniform_knot_vec(self,num_cps,order):
        knot_vector = np.zeros(num_cps + order)
        for i in range(num_cps + order):
            knot_vector[i] = (i - order + 1) / (num_cps - order + 1)
        return knot_vector

    def get_bases(self):
        surf_000   = self.Volume.get_basis_matrix(self.u['surf'],self.v['surf'],self.w['surf'],0,0,0)
        surf_100   = self.Volume.get_basis_matrix(self.u['surf'],self.v['surf'],self.w['surf'],1,0,0)
        surf_010   = self.Volume.get_basis_matrix(self.u['surf'],self.v['surf'],self.w['surf'],0,1,0)
        surf_001   = self.Volume.get_basis_matrix(self.u['surf'],self.v['surf'],self.w['surf'],0,0,1)
        bases_surf = np.stack((surf_000,surf_100,surf_010,surf_001))
        hess_200  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],2,0,0)
        hess_020  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],0,2,0)
        hess_110  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],1,1,0)
        hess_011  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],0,1,1)
        hess_101  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],1,0,1)
        hess_002  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],0,0,2)
        bases_curv = np.stack((hess_200,hess_110,hess_020,hess_101,hess_011,hess_002))
        return bases_surf, bases_curv

    def init_cps_Hicken(self,k=10,rho=10):
        rangex = self.dimensions[0]
        rangey = self.dimensions[1]
        rangez = self.dimensions[2]
        # Order 3, index 0.5: basis = [1/2, 1/2]
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
        A = np.array([[self.num_cps[2]-1-Q, Q],
                    [Q, self.num_cps[2]-1-Q]])
        b = np.array([rangez[0]*(self.num_cps[2]-1), rangez[1]*(self.num_cps[2]-1)])
        zn = np.linalg.solve(A,b)

        cps = np.zeros((np.product(self.num_cps), 4))
        cps[:, 0] = np.einsum('i,j,k->ijk', np.linspace(xn[0],xn[1],self.num_cps[0]), np.ones(self.num_cps[1]),np.ones(self.num_cps[2])).flatten()
        cps[:, 1] = np.einsum('i,j,k->ijk', np.ones(self.num_cps[0]), np.linspace(yn[0],yn[1],self.num_cps[1]),np.ones(self.num_cps[2])).flatten()
        cps[:, 2] = np.einsum('i,j,k->ijk', np.ones(self.num_cps[0]), np.ones(self.num_cps[1]),np.linspace(zn[0],zn[1],self.num_cps[2])).flatten()

        dataset = KDTree(self.exact[0])
        distances,indices = dataset.query(cps[:,0:3],k=k)
        d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
        exp = np.exp(-rho*d_norm)
        Dx = dataset.data[indices] - np.reshape(cps[:,0:3],(np.product(self.num_cps),1,3))
        phi = np.einsum('ijk,ijk,ij,i->i',Dx,self.exact[1][indices],exp,1/np.sum(exp,axis=1))
        np.random.seed(1)
        phi += 1e-3*self.Bbox_diag*(2*np.random.rand(np.product(self.num_cps))-1)
        cps[:, 3] = phi/self.Bbox_diag
        return cps

    def get_basis(self,loc='surf',du=0,dv=0,dw=0):
        basis = self.Volume.get_basis_matrix(self.u[loc],self.v[loc],self.w[loc],du,dv,dw)
        return basis

    def eval_pts(self,pts):
        u,v,w = self.spatial_to_parametric(pts)
        b = self.Volume.get_basis_matrix(u,v,w,0,0,0)
        return b.dot(self.cps[:,3])

    def eval_surface(self):
        u,v,w = self.spatial_to_parametric(self.exact[0])
        b = self.Volume.get_basis_matrix(u,v,w,0,0,0)
        return b.dot(self.cps[:,3])

    def check_local_RMS_error(self,bbox_perc,res,num_samp=None):
        if num_samp is None:
            num_samp = len(self.surf_pts)
        ep_max = bbox_perc*self.Bbox_diag / 100
        ep_range = np.linspace(-ep_max,ep_max,res)
        dataset = KDTree(self.exact[0])
        RMS_local = np.zeros(len(ep_range))
        np.random.seed(1)
        rng = np.random.default_rng()
        indx = rng.choice(np.size(self.exact[0],0), size=num_samp, replace=False)
        sample_pts = self.exact[0][indx,:]
        sample_normals = self.exact[1][indx,:]
        for i,ep in enumerate(ep_range):
            i_pts = sample_pts + ep*sample_normals
            u,v,w = self.spatial_to_parametric(i_pts)
            b = self.Volume.get_basis_matrix(u,v,w,0,0,0)
            phi = b.dot(self.cps[:,3])
            phi_ex,_ = dataset.query(i_pts,k=1)
            RMS_local[i] = np.sqrt(np.mean( (abs(phi)-phi_ex)**2  ))
        return np.linspace(-bbox_perc,bbox_perc,res), RMS_local/self.Bbox_diag

    def check_local_max_error(self,bbox_perc,res,num_samp=None):
        from utils.Hicken_Kaur import KS_eval
        if num_samp is None:
            num_samp = len(self.surf_pts)
        ep_max = bbox_perc*self.Bbox_diag / 100
        ep_range = np.linspace(-ep_max,ep_max,res)
        dataset = KDTree(self.exact[0])
        MAX_local = np.zeros(len(ep_range))
        np.random.seed(1)
        rng = np.random.default_rng()
        indx = rng.choice(np.size(self.exact[0],0), size=num_samp, replace=False)
        sample_pts = self.exact[0][indx,:]
        sample_normals = self.exact[1][indx,:]
        for i,ep in enumerate(ep_range):
            i_pts = sample_pts + ep*sample_normals
            u,v,w = self.spatial_to_parametric(i_pts)
            b = self.Volume.get_basis_matrix(u,v,w,0,0,0)
            phi = b.dot(self.cps[:,3])
            phi_ex,_ = dataset.query(i_pts,k=1)
            MAX_local[i] = np.max(abs(abs(phi)-phi_ex))
        return np.linspace(-bbox_perc,bbox_perc,res), MAX_local/self.Bbox_diag

    def check_local_RMS_error_via_hicken(self,bbox_perc,res,num_samp=None):
        from utils.Hicken_Kaur import KS_eval
        if num_samp is None:
            num_samp = len(self.surf_pts)
        ep_max = bbox_perc*self.Bbox_diag / 100
        ep_range = np.linspace(-ep_max,ep_max,res)
        dataset = KDTree(self.exact[0])
        RMS_local = np.zeros(len(ep_range))
        np.random.seed(1)
        rng = np.random.default_rng()
        indx = rng.choice(np.size(self.exact[0],0), size=num_samp, replace=False)
        sample_pts = self.exact[0][indx,:]
        sample_normals = self.exact[1][indx,:]
        for i,ep in enumerate(ep_range):
            i_pts = sample_pts + ep*sample_normals
            u,v,w = self.spatial_to_parametric(i_pts)
            b = self.Volume.get_basis_matrix(u,v,w,0,0,0)
            phi = b.dot(self.cps[:,3])
            phi_ex = KS_eval(i_pts,dataset,self.exact[1],1,1)
            RMS_local[i] = np.sqrt(np.mean( (phi-phi_ex)**2  ))
        return np.linspace(-bbox_perc,bbox_perc,res), RMS_local/self.Bbox_diag

    def check_local_max_error_via_hicken(self,bbox_perc,res,num_samp=None):
        from utils.Hicken_Kaur import KS_eval
        if num_samp is None:
            num_samp = len(self.surf_pts)
        ep_max = bbox_perc*self.Bbox_diag / 100
        ep_range = np.linspace(-ep_max,ep_max,res)
        dataset = KDTree(self.exact[0])
        MAX_local = np.zeros(len(ep_range))
        np.random.seed(1)
        rng = np.random.default_rng()
        indx = rng.choice(np.size(self.exact[0],0), size=num_samp, replace=False)
        sample_pts = self.exact[0][indx,:]
        sample_normals = self.exact[1][indx,:]
        for i,ep in enumerate(ep_range):
            i_pts = sample_pts + ep*sample_normals
            u,v,w = self.spatial_to_parametric(i_pts)
            b = self.Volume.get_basis_matrix(u,v,w,0,0,0)
            phi = b.dot(self.cps[:,3])
            phi_ex = KS_eval(i_pts,dataset,self.exact[1],1,1)
            MAX_local[i] = np.max(abs(abs(phi)-phi_ex))
        return np.linspace(-bbox_perc,bbox_perc,res), MAX_local/self.Bbox_diag

    def visualize_current(self,res=30):
        gold = (198/255, 146/255, 20/255)
        sns.set()
        plt.figure()
        ax = plt.axes(projection='3d')
        x = self.dimensions[0]
        y = self.dimensions[1]
        z = self.dimensions[2]
        u = np.einsum('i,j,k->ijk', np.linspace(0,1,res), np.ones(res),np.ones(res)).flatten()
        v = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(0,1,res),np.ones(res)).flatten()
        w = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(0,1,res)).flatten()
        basis = self.Volume.get_basis_matrix(u, v, w, 0, 0, 0)
        phi = basis.dot(self.cps[:,3]).reshape((res,res,res))
        verts, faces,_,_ = marching_cubes(phi, 0)
        verts = verts*np.diff(self.dimensions).flatten()/(res-1) + self.dimensions[:,0]
        level_set = Poly3DCollection(verts[faces],linewidth=0.25,alpha=1,facecolor=gold,edgecolor='k')
        ax.add_collection3d(level_set)
        ax.plot(self.surf_pts[:,0],self.surf_pts[:,1],self.surf_pts[:,2],
                'k.',label='surface points')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title('Current Level Set $n_{\Gamma}$=%i'%len(self.surf_pts))
        ax.set_xticks([x[0],(x[1]+x[0])/2,x[1]])
        ax.set_yticks([y[0],(y[1]+y[0])/2,y[1]])
        ax.set_zticks([z[0],(z[1]+z[0])/2,z[1]])
        center = np.mean(self.dimensions,axis=1)
        d = np.max(np.diff(self.dimensions,axis=1))
        ax.set_xlim(center[0]-d/2, center[0]+d/2)
        ax.set_ylim(center[1]-d/2, center[1]+d/2)
        ax.set_zlim(center[2]-d/2, center[2]+d/2)
        # ax.axis('equal')

        plt.figure()
        ax = plt.axes()
        res = 200
        ones = np.ones(res)
        diag = np.linspace(0,1,res)
        basis = self.Volume.get_basis_matrix(diag, 0.5*ones, 0.5*ones, 0, 0, 0)
        pts = basis.dot(self.cps[:,3])
        ax.plot(diag, pts, '-', label='X-axis')
        basis = self.Volume.get_basis_matrix(0.5*ones, diag, 0.5*ones, 0, 0, 0)
        pts = basis.dot(self.cps[:,3])
        ax.plot(diag, pts, '-', label='Y-axis')
        basis = self.Volume.get_basis_matrix(0.5*ones, 0.5*ones, diag, 0, 0, 0)
        pts = basis.dot(self.cps[:,3])
        ax.plot(diag, pts, '-', label='Z-axis')
        ax.axis([0,1,-d,d])
        ax.set_xticks([0,0.5,1])
        ax.set_yticks([-d,0,d])
        ax.set_xlabel('Normalized Location')
        ax.set_ylabel('Phi')
        ax.set_title('Phi along 1D slices')
        ax.legend()
        return
