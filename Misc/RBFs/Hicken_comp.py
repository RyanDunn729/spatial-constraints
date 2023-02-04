import numpy as np
from scipy.spatial import KDTree
from openmdao.api import ExplicitComponent

class Hicken_comp(ExplicitComponent):
    def initialize(self):
        self.options.declare('norm_vec')
        self.options.declare('KDTree')
        self.options.declare('rho')
        self.options.declare('num_pts',types=int)
        self.options.declare('k',types=int)

    def setup(self):
        num_pts = self.options['num_pts']
        self.add_input('pt', shape=(num_pts,3))
        self.add_output('signedfun', shape=(num_pts,))

    def setup_partials(self):
        num_pts = self.options['num_pts']
        row_ind = np.empty(3*num_pts)
        for i in range(num_pts):
            row_ind[3*i : 3*(i+1)] = i
        col_ind = np.arange(3*num_pts)
        self.declare_partials(of='signedfun', wrt='pt',
                              rows=row_ind,cols=col_ind) #,method='fd')

    def compute(self, inputs, outputs):
        norm_vec = self.options['norm_vec']
        num_pts = self.options['num_pts']
        KDTree = self.options['KDTree']
        rho = self.options['rho']
        k = self.options['k']
        pts = inputs['pt']

        distances,indices = KDTree.query(pts,k=k)
        d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
        exp = np.exp(-rho*d_norm)
        Dx = KDTree.data[indices] - np.reshape(pts,(num_pts,1,3))
        phi = np.einsum('ijk,ij->i',Dx*norm_vec[indices],exp)/np.sum(exp,axis=1)

        outputs['signedfun'] = phi
    
    def compute_partials(self, inputs, partials):
        norm_vec = self.options['norm_vec']
        num_pts = self.options['num_pts']
        KDTree = self.options['KDTree']
        rho = self.options['rho']
        k = self.options['k']
        pts = inputs['pt']

        distances,indices = KDTree.query(pts,k=k)
        di = KDTree.data[indices] - pts.reshape(num_pts,1,3)
        check = 2*np.heaviside(np.einsum('ijk,ijk->ij',di,norm_vec[indices]),1) - 1
        sign = 2*np.heaviside(np.sum(check,axis=1),1) - 1

        d_norm = (distances.T - distances[:,0]).T
        exp = np.exp(-rho*d_norm)

        dhi = np.empty((len(pts),3))
        dlow = np.empty((len(pts),3))
        for i,(i_pt,ind) in enumerate(zip(pts,indices)):
            k_pts = KDTree.data[ind]

            dx = np.transpose((i_pt-k_pts).T/(distances[i] + 1e-20)) # Avoid dividing by zero when distance = 0

            dexp = dx-np.repeat(dx[0],k).reshape(3,k).T
            hi_terms = dx - rho*np.einsum('i,ij->ij',distances[i],dexp)
            dhi[i] = np.einsum('ij,i->j',hi_terms,exp[i])
            dlow[i] = np.einsum('ij,i->j',-rho*dexp,exp[i])

        low = np.sum(exp,axis=1)
        hi = np.einsum('ij,ij->i',distances,exp)
        
        # Quotient Rule (OLD)
        deriv = np.einsum('i,ij->ij', 1/low**2, (np.einsum('i,ij->ij',sign*low,dhi) - np.einsum('i,ij->ij',sign*hi,dlow)))

        partials['signedfun','pt'] = deriv.flatten()

def deriv_eval(norm_vec, num_pts, KDTree, rho, k, pts):

        distances,indices = KDTree.query(pts,k=k)
        di = KDTree.data[indices] - pts.reshape(num_pts,1,3)
        check = 2*np.heaviside(np.einsum('ijk,ijk->ij',di,norm_vec[indices]),1) - 1
        sign = 2*np.heaviside(np.sum(check,axis=1),1) - 1

        d_norm = (distances.T - distances[:,0]).T
        exp = np.exp(-rho*d_norm)

        dhi = np.empty((len(pts),3))
        dlow = np.empty((len(pts),3))
        for i,(i_pt,ind) in enumerate(zip(pts,indices)):
            k_pts = KDTree.data[ind]

            dx = np.transpose((i_pt-k_pts).T/(distances[i] + 1e-20)) # Avoid dividing by zero when distance = 0

            dexp = dx-np.repeat(dx[0],k).reshape(3,k).T
            hi_terms = dx - rho*np.einsum('i,ij->ij',distances[i],dexp)
            dhi[i] = np.einsum('ij,i->j',hi_terms,exp[i])
            dlow[i] = np.einsum('ij,i->j',-rho*dexp,exp[i])

        low = np.sum(exp,axis=1)
        hi = np.einsum('ij,ij->i',distances,exp)
        
        # Quotient Rule (OLD)
        deriv = np.einsum('i,ij->ij', 1/low**2, (np.einsum('i,ij->ij',sign*low,dhi) - np.einsum('i,ij->ij',sign*hi,dlow)))

        return deriv.flatten()

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    import matplotlib.pyplot as plt
    from scipy.spatial import KDTree
    from stl.mesh import Mesh
    import numpy as np

    # How many neighbors to sample nearby
    k = 20
    # weighting parameter, increases accuracy as -> inf
    rho = 20
    # Number of backbone points to sample
    pts_x = 35
    pts_y = 35
    pts_z = 35
    num_backbone_pts = pts_x*pts_y*pts_z

    filename = 'stl-files/heart_case03.stl'           # 1888 midpts, 5664 vertices
    # filename = 'stl-files/Stanford_bunny_fine.stl'    # 270021 midpts, 810063 vertices
    # filename = 'stl-files/stitch.stl'                 # 509580 midpts, 1528740 vertices
    # filename = 'stl-files/trachea_1191v.stl'          # 2197 midpts, 6591 vertices
    # filename = 'stl-files/full_body.stl'              # 3652 midpts, 10956 vertices
    # filename = 'stl-files/scan_mesh.stl'              # 255830 midpts, 767490 vertices
    # filename = 'stl-files/Dragon_pro.stl'             # 359224 midpts, 1077672 vertices

    mesh_import = Mesh.from_file(filename)
    temp = mesh_import.points.reshape(len(mesh_import.points),3,3)
    midpts = np.mean(temp,axis=1)
    # Calculate the location of the centroid of each triangle
    temp = mesh_import.points.reshape(3*len(mesh_import.points),3)
    mesh_pts = np.unique(temp,axis=0)
    norm_vec = mesh_import.get_unit_normals()

    # Normal vectors are in-line with the centroids of each triangle
    dataset = KDTree(midpts)

    # nodes: Mesh points
    # normals: Average normal vector from nearest 'k' centroids
    _,indices = dataset.query(mesh_pts,k=3)
    avg_norm_vec = np.mean(norm_vec[indices],axis=1)
    new_norm_vec = np.einsum('ij,i->ij',avg_norm_vec, 1/np.linalg.norm(avg_norm_vec,axis=1))
    # norm_vec = new_norm_vec
    # dataset = KDTree(mesh_pts)
    
    # Sample backbone points, evenly distributed around mesh with a 10% border
    border = 0.10
    minx,miny,minz = mesh_import.min_
    maxx,maxy,maxz = mesh_import.max_
    minx -= ((maxx-minx)*border)
    maxx += ((maxx-minx)*border)
    miny -= ((maxy-miny)*border)
    maxy += ((maxy-miny)*border)
    minz -= ((maxz-minz)*border)
    maxz += ((maxz-minz)*border)
    pt_grid = np.empty((pts_x * pts_y * pts_z, 3))
    pt_grid[:, 0] = np.einsum('i,j,k->ijk', np.linspace(minx,maxx,pts_x), np.ones(pts_y), np.ones(pts_z)).flatten()
    pt_grid[:, 1] = np.einsum('i,j,k->ijk', np.ones(pts_x), np.linspace(miny,maxy,pts_y), np.ones(pts_z)).flatten()
    pt_grid[:, 2] = np.einsum('i,j,k->ijk', np.ones(pts_x), np.ones(pts_y), np.linspace(minz,maxz,pts_z)).flatten()

    group = Group()
    comp = Hicken_comp(KDTree=dataset,norm_vec=norm_vec,num_pts=num_backbone_pts,k=k,rho=rho)
    group.add_subsystem('Hicken_comp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group

    prob.setup()
    
    prob['pt'] = pt_grid

    # prob.check_partials(compact_print=False)
    prob.run_model()

    print(prob['pt'][0:35,2])
    print(prob['signedfun'])

    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    # prob.check_partials(compact_print=True)
    def plot3d(pts,title=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(pts[:,0],pts[:,1],pts[:,2])
        if title:
            ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    # plot3d(pt_grid[prob['signedfun']>0])

    plt.plot(prob['pt'][0:35,2],prob['signedfun'][0:35],'k-')

    fd_deriv = np.diff(prob['signedfun'][0:35])/np.diff(prob['pt'][0:35,2])
    plt.plot(prob['pt'][1:35,2],fd_deriv,'--')

    deriv = deriv_eval(norm_vec, 35, dataset, rho, k, prob['pt'][0:35,:])
    dxyz = np.reshape(deriv,(35,3))
    plt.plot(prob['pt'][0:35,2],dxyz[:,2],'-')
    plt.show()