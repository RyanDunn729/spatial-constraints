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

    def __init__(self, exact, surf_pts, normals, max_cps, R, border, order):
        k = 10 # Num of nearest points to garuntee interior or exterior point
        self.order = order # Must be >2
        self.u = {}
        self.v = {}
        self.w = {}
        self.exact = exact
        self.surf_pts = surf_pts
        self.normals = normals

        # Bounding Box of Exact Mesh
        lower = np.min(exact[0],axis=0)
        upper = np.max(exact[0],axis=0)
        diff = upper-lower
        self.Bbox_diag = np.linalg.norm(diff)
        self.dimensions = np.stack((lower-diff*border, upper+diff*border),axis=1)
        self.V = np.product(np.diff(self.dimensions))
        dxyz = np.diff(self.dimensions).flatten()/self.Bbox_diag

        # Scale the resolutions of cps and hess samples
        frac = dxyz / np.max(dxyz)
        num_cps = np.zeros(3,dtype=int)
        num_hess = np.zeros(3,dtype=int)
        for i in range(3):
            if frac[i] < 0.6:
                frac[i] = 0.6
            num_cps[i] = int(np.round(frac[i]*max_cps)+order-1)
            num_hess[i] = int(np.round(R*(num_cps[i]-order+R-1)-1))
        self.num_cps = num_cps
        # Get initial control points
        self.cps = self.init_cps_Hicken(k=k,rho=10*len(exact[0]))

        # Standard uniform knot vectors
        kv_u = self.std_uniform_knot_vec(num_cps[0], order)
        kv_v = self.std_uniform_knot_vec(num_cps[1], order)
        kv_w = self.std_uniform_knot_vec(num_cps[2], order)
        # Define Bspline Volume object
        self.Volume = BSplineVolume('name',order,order,order,kv_u,kv_v,kv_w,num_cps,self.cps)

        # Get uvw_mesh
        self.u['surf'], self.v['surf'], self.w['surf'] = self.spatial_to_parametric(surf_pts)
        # Get uvw_hess
        self.v['hess'], self.u['hess'], self.w['hess'] = np.meshgrid(
            np.linspace(0,1, num_hess[1]+2)[1:num_hess[1]+1],
            np.linspace(0,1, num_hess[0]+2)[1:num_hess[0]+1],
            np.linspace(0,1, num_hess[2]+2)[1:num_hess[2]+1])
        self.u['hess'] = self.u['hess'].flatten()
        self.v['hess'] = self.v['hess'].flatten()
        self.w['hess'] = self.w['hess'].flatten()

        # Sizing
        self.dV = self.V/np.product(num_hess)
        # Get scaling matrix
        self.scaling = 1/dxyz

        self.num_cps_pts = int(np.product(num_cps))
        self.num_hess_pts = int(np.product(num_hess))

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
        return self.scaling, self.dV, self.V, surf, curv

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
        del self.u, self.v, self.w
        return bases_surf, bases_curv

    def init_cps_Hicken(self,k=10,rho=1000):
        rangex = self.dimensions[0]
        rangey = self.dimensions[1]
        rangez = self.dimensions[2]
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
        phi += 1e-14*(2*np.random.rand(np.product(self.num_cps))-1)
        cps[:, 3] = phi/self.Bbox_diag
        return cps

    def eval_pts(self,pts):
        u,v,w = self.spatial_to_parametric(pts)
        b = self.Volume.get_basis_matrix(u,v,w,0,0,0)
        return b.dot(self.cps[:,3])

    def eval_surface(self):
        u,v,w = self.spatial_to_parametric(self.exact[0])
        b = self.Volume.get_basis_matrix(u,v,w,0,0,0)
        return b.dot(self.cps[:,3])

    def check_local_RMS_error(self,bbox_perc,res,num_samp=1000):
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

    def get_energy_terms(self,Prob):
        L = Prob['lambdas']
        E = np.zeros(3)
        num_surf = len(self.surf_pts)
        E[2] = L[2]*np.sum(Prob['phi_surf']**2)/num_surf
        E[1] = L[1]*np.sum((Prob['dpdx_surf']+self.normals[:,0])**2)/num_surf
        E[1] += L[1]*np.sum((Prob['dpdy_surf']+self.normals[:,1])**2)/num_surf
        E[1] += L[1]*np.sum((Prob['dpdz_surf']+self.normals[:,2])**2)/num_surf
        E[0] = L[0]*self.dV/self.V*np.sum(Prob['Fnorm']**2)/self.num_hess_pts

        E_scaled = np.zeros(3)
        E_scaled[0] = E[0]/L[0]
        E_scaled[1] = E[1]/L[1]
        E_scaled[2] = E[2]/L[2]
        return E, E_scaled

    def visualize_current(self,res):
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

    def get_RMS_Fnorm(self):
        init_cps = self.init_cps_Hicken()
        num_hess = np.zeros(3,dtype=int)
        R = 2
        for i in range(3):
            num_hess[i] = int(np.round(R*(self.num_cps[i]-self.order+R-1)-1))
        # Get uvw_hess
        self.u = {}
        self.v = {}
        self.w = {}
        self.v['hess'], self.u['hess'], self.w['hess'] = np.meshgrid(
            np.linspace(0,1, num_hess[1]+2)[1:num_hess[1]+1],
            np.linspace(0,1, num_hess[0]+2)[1:num_hess[0]+1],
            np.linspace(0,1, num_hess[2]+2)[1:num_hess[2]+1])
        self.u['hess'] = self.u['hess'].flatten()
        self.v['hess'] = self.v['hess'].flatten()
        self.w['hess'] = self.w['hess'].flatten()
        hess_200  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],2,0,0)
        hess_020  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],0,2,0)
        hess_110  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],1,1,0)
        hess_011  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],0,1,1)
        hess_101  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],1,0,1)
        hess_002  = self.Volume.get_basis_matrix(self.u['hess'],self.v['hess'],self.w['hess'],0,0,2)
        bases_curv = np.stack((hess_200,hess_110,hess_020,hess_101,hess_011,hess_002))

        scaling = self.scaling
        bases = bases_curv

        init_cps[:,3] *= self.Bbox_diag
        dp_dxx = scaling[0]*scaling[0]*bases[0].dot(init_cps[:,3])
        dp_dxy = scaling[0]*scaling[1]*bases[1].dot(init_cps[:,3])
        dp_dyy = scaling[1]*scaling[1]*bases[2].dot(init_cps[:,3])
        dp_dxz = scaling[0]*scaling[2]*bases[3].dot(init_cps[:,3])
        dp_dyz = scaling[1]*scaling[2]*bases[4].dot(init_cps[:,3])
        dp_dzz = scaling[2]*scaling[2]*bases[5].dot(init_cps[:,3])
        hess = np.zeros((int(np.product(num_hess)),3,3))
        hess[:,0,0] = dp_dxx
        hess[:,1,0] = dp_dxy
        hess[:,0,1] = dp_dxy
        hess[:,1,1] = dp_dyy
        hess[:,0,2] = dp_dxz
        hess[:,2,0] = dp_dxz
        hess[:,1,2] = dp_dyz
        hess[:,2,1] = dp_dyz
        hess[:,2,2] = dp_dzz
        initial = np.sqrt(np.mean(  np.linalg.norm(hess,axis=(1,2),ord='fro')**2 ))

        dp_dxx = scaling[0]*scaling[0]*bases[0].dot(self.cps[:,3])
        dp_dxy = scaling[0]*scaling[1]*bases[1].dot(self.cps[:,3])
        dp_dyy = scaling[1]*scaling[1]*bases[2].dot(self.cps[:,3])
        dp_dxz = scaling[0]*scaling[2]*bases[3].dot(self.cps[:,3])
        dp_dyz = scaling[1]*scaling[2]*bases[4].dot(self.cps[:,3])
        dp_dzz = scaling[2]*scaling[2]*bases[5].dot(self.cps[:,3])
        hess = np.zeros((int(np.product(num_hess)),3,3))
        hess[:,0,0] = dp_dxx
        hess[:,1,0] = dp_dxy
        hess[:,0,1] = dp_dxy
        hess[:,1,1] = dp_dyy
        hess[:,0,2] = dp_dxz
        hess[:,2,0] = dp_dxz
        hess[:,1,2] = dp_dyz
        hess[:,2,1] = dp_dyz
        hess[:,2,2] = dp_dzz
        final = np.sqrt(np.mean( np.linalg.norm(hess,axis=(1,2),ord='fro')**2 ))
        return final/initial
