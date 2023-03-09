import numpy as np
from openmdao.api import ExplicitComponent

class PtComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('num_pt', default=2, types=int)
        self.options.declare('num_pt', default=3, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('p') # mesh points
        self.options.declare('normals') # Normals of each mesh points
        
    def setup(self):
        num_pt = self.options['num_pt']
        normals = self.options['normals']
        k = self.options['k']
        p = self.options['p'] 
        #Inputs
        self.add_input('pt', shape=(num_pt,3)) # the points you want to compute


        # outputs
        # (sample_pts,k=1,num_mesh_pts,3)
        # swap (num_pt,k=1) => (k=1,num_pt)
        self.add_output('norm_vec',shape=(num_pt,k,p.shape[0],3))
        # (sample_pts,k=1,num_mesh_pts)
        self.add_output('distance',shape=(num_pt,k,p.shape[0]))


        # partials
        row_indices = np.outer(np.arange(0,num_pt*p.shape[0]),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(num_pt),np.outer(np.ones(p.shape[0]),np.array([0,1,2])).flatten()) \
            + (np.arange(0,num_pt*3,3).reshape(-1,1))
        self.declare_partials('distance', 'pt',rows=row_indices,cols=col_indices.flatten())
        row_indices_n = np.outer(np.arange(0,num_pt*p.shape[0]*3),np.ones(3)).flatten()
        col_indices_n = np.outer(np.ones(num_pt),np.outer(np.ones(p.shape[0]*3),np.array([0,1,2])).flatten()) \
            + (np.arange(0,num_pt*3,3).reshape(-1,1))
        self.declare_partials('norm_vec', 'pt',rows=row_indices_n,cols=col_indices_n.flatten())
             
    def compute(self,inputs,outputs):

        num_pt = self.options['num_pt']
        normals = self.options['normals']
        p = self.options['p']
        pt = inputs['pt']

        vec = np.zeros((num_pt,1,p.shape[0],3))
        norm_vec = np.zeros((num_pt,1,p.shape[0],3))
        
        # compute the distance btw ctr points and mesh points
        for i in range(num_pt):
            vec[i,:,:] = pt[i,:] -  p
        dis = np.linalg.norm(vec,axis=3)
        epsilon = 1e-8
        
        # normalize the vector
        norm_vec[:,:,:,0] = vec[:,:,:,0] / (dis+epsilon)
        norm_vec[:,:,:,1] = vec[:,:,:,1] / (dis+epsilon)
        norm_vec[:,:,:,2] = vec[:,:,:,2] / (dis+epsilon)
        self.vec = vec
        self.dis = dis

        # using the formulation of the anatomical constriaints
        dis = np.sum((vec)**2,axis=3)**0.125
        outputs['norm_vec'] = norm_vec
        outputs['distance'] = dis
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        
        p = self.options['p']
        num_pt = self.options['num_pt']
        vec = self.vec
        dis = self.dis
        
        # partial p
        Peu_ppt = np.zeros((num_pt,1,p.shape[0],3))
        Peu_ppt[:,:,:,0] = (np.sum((vec)**2,axis=3) **-0.875)*vec[:,:,:,0]*0.25
        Peu_ppt[:,:,:,1] = (np.sum((vec)**2,axis=3) **-0.875)*vec[:,:,:,1]*0.25
        Peu_ppt[:,:,:,2] = (np.sum((vec)**2,axis=3) **-0.875)*vec[:,:,:,2]*0.25
        
        Pnd_ppt = np.zeros((num_pt,1,p.shape[0],3,3))
        Pnd_ppt[:,:,:,0,0] =  1/dis + vec[:,:,:,0] * -0.5*(np.sum(vec**2,3)**-1.5) * 2 * vec[:,:,:,0]
        Pnd_ppt[:,:,:,1,1] =  1/dis + vec[:,:,:,1] * -0.5*(np.sum(vec**2,3)**-1.5) * 2 * vec[:,:,:,1]
        Pnd_ppt[:,:,:,2,2] =  1/dis + vec[:,:,:,2] * -0.5*(np.sum(vec**2,3)**-1.5) * 2 * vec[:,:,:,2]
        Pnd_ppt[:,:,:,0,1] =  -0.5*vec[:,:,:,0]*(np.sum(vec**2,3)**-1.5) * 2 * vec[:,:,:,1]
        Pnd_ppt[:,:,:,0,2] =  -0.5*vec[:,:,:,0]*(np.sum(vec**2,3)**-1.5) * 2 * vec[:,:,:,2]
        Pnd_ppt[:,:,:,1,0] =  -0.5*vec[:,:,:,1]*(np.sum(vec**2,3)**-1.5) * 2 * vec[:,:,:,0]
        Pnd_ppt[:,:,:,1,2] =  -0.5*vec[:,:,:,1]*(np.sum(vec**2,3)**-1.5) * 2 * vec[:,:,:,2]
        Pnd_ppt[:,:,:,2,0] =  -0.5*vec[:,:,:,2]*(np.sum(vec**2,3)**-1.5) * 2 * vec[:,:,:,0]
        Pnd_ppt[:,:,:,2,1] =  -0.5*vec[:,:,:,2]*(np.sum(vec**2,3)**-1.5) * 2 * vec[:,:,:,1]

        partials['norm_vec','pt'][:] = Pnd_ppt.flatten()
        partials['distance','pt'][:] = Peu_ppt.flatten()

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    
    group = Group()
    num_pt = 10
    
    comp = IndepVarComp()
    # comp.add_output('tube_section_length', val=0.1 * np.ones((2,3)))
    # comp.add_output('beta', val=0.1 * np.ones((2,3)))
    p = np.random.rand(30,3)
    normals = np.random.rand(30,3)

    comp.add_output('pt', val = np.random.random((num_pt,3))*10)
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
        
    comp = PtComp(num_pt=num_pt,p=p,normals=normals)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    