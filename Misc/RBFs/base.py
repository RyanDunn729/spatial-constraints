import numpy as np
from lsdo_geo.bsplines.bspline_volume import BSplineVolume
from handy_funcs import std_uniform_knot_vec, import_mesh, sdf_sample_points, init_cps_constant
from handy_funcs import plot3d
class MyProblem(object):

    def __init__(self, filename, num_samples, num_cps, num_hess, border, max_scale):
        max_iter = 350 # max iterations in Bsp_vol projection
        guess_grid = 0 # Size sample grid for projection initialization
        k = 3 # Num of nearest points to garuntee interior or exterior point
        order = 4 # Must be 4
        self.u = {}
        self.v = {}
        self.w = {}

        # Import mesh
        mesh_import, mesh_pts = import_mesh(filename)

        # Add border region
        minx,miny,minz = mesh_import.min_
        maxx,maxy,maxz = mesh_import.max_
        dx = maxx-minx
        dy = maxy-miny
        dz = maxz-minz
        print('mesh_x: [',minx,', ',maxx,']')
        print('mesh_y: [',miny,', ',maxy,']')
        print('mesh_z: [',minz,', ',maxz,']\n')
        self.x = [minx-(dx*border), maxx+(dx*border)]
        self.y = [miny-(dy*border), maxy+(dy*border)]
        self.z = [minz-(dz*border), maxz+(dz*border)]
        print('adjusted_mesh_x: ',self.x)
        print('adjusted_mesh_y: ',self.y)
        print('adjusted_mesh_z: ',self.z,'\n')

        # Scale the resolutions of cps, hess, and sample points
        frac = [dx,dy,dz] / np.min([dx,dy,dz])
        print('Scaling Ratios: ',frac)
        if len(num_cps)==1:
            temp = num_cps[0]
            num_cps = np.empty(3,dtype=int)
            for i in range(3):
                num_cps[i] = int(np.min([temp*frac[i], max_scale*temp]))
        if len(num_hess)==1:
            temp = num_hess[0]
            num_hess = np.empty(3,dtype=int)
            for i in range(3):
                num_hess[i] = int(np.min([temp*frac[i], max_scale*temp]))
        print('num_hess:    ',num_hess)
        print('num_cps:     ',num_cps)
        print('num_samples: ',num_samples,'\n')

        # Get initial control points
        self.cps = init_cps_constant(num_cps,order,self.x,self.y,self.z,val=-1)
        print('phi0_min: ',np.min(self.cps[:,3]))
        print('phi0_max: ',np.max(self.cps[:,3]),'\n')

        # Standard uniform knot vectors
        knot_vec_u = std_uniform_knot_vec(num_cps[0], order)
        knot_vec_v = std_uniform_knot_vec(num_cps[1], order)
        knot_vec_w = std_uniform_knot_vec(num_cps[2], order)

        # Define Bspline Volume object
        self.Volume = BSplineVolume('name',order,order,order,knot_vec_u,knot_vec_v,knot_vec_w,num_cps,self.cps)

        # Get uvw_mesh
        self.u['mesh'], self.v['mesh'], self.w['mesh'] = self.Volume.project(mesh_pts,max_iter,guess_grid)
        # Get uvw_hess
        hess_pts = np.empty((num_hess[0] * num_hess[1] * num_hess[2], 3))
        hess_pts[:, 0] = np.einsum('i,j,k->ijk', np.linspace(self.x[0], self.x[1], num_hess[0]+2)[1:num_hess[0]+1], np.ones(num_hess[1]),np.ones(num_hess[2])).flatten()
        hess_pts[:, 1] = np.einsum('i,j,k->ijk', np.ones(num_hess[0]), np.linspace(self.y[0], self.y[1], num_hess[1]+2)[1:num_hess[1]+1],np.ones(num_hess[2])).flatten()
        hess_pts[:, 2] = np.einsum('i,j,k->ijk', np.ones(num_hess[0]), np.ones(num_hess[1]),np.linspace(self.z[0], self.z[1], num_hess[2]+2)[1:num_hess[2]+1]).flatten()
        self.u['hess'], self.v['hess'], self.w['hess'] = self.Volume.project(hess_pts,max_iter,guess_grid)
        # Evaluate samples and label them interior or exterior
        inside_pts, outside_pts, mid_pts, norm_vec = sdf_sample_points(filename,num_samples,k=k)
        self.save_norm_vec = norm_vec
        self.u['int'], self.v['int'], self.w['int'] = self.Volume.project(inside_pts,max_iter,guess_grid)
        self.u['ext'], self.v['ext'], self.w['ext'] = self.Volume.project(outside_pts,max_iter,guess_grid)
        self.u['mid'], self.v['mid'], self.w['mid'] = self.Volume.project(mid_pts,max_iter,guess_grid)

        # Sizing
        self.V = np.diff(self.x) * np.diff(self.y) * np.diff(self.z)
        self.dV = self.V/(num_hess[0] * num_hess[1] * num_hess[2])
        # Get scaling matrix
        self.inv_scaling_matrix = np.linalg.inv(np.diag([dx*(1+2*border),dy*(1+2*border),dz*(1+2*border)]))
        print('dxdu: ',dx*(1+2*border))
        print('dydv: ',dy*(1+2*border))
        print('dzdw: ',dz*(1+2*border),'\n')

    def get_values(self):
        return self.Volume, self.u, self.v, self.w, self.inv_scaling_matrix, self.dV, self.V, self.save_norm_vec

    def set_cps(self, cps_phi):
        self.cps[:,3] = cps_phi
        self.Volume.control_points = self.cps
        return