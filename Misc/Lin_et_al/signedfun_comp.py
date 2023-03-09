import numpy as np
from openmdao.api import ExplicitComponent

class SignedfunComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('k',default=1, types=int)
        self.options.declare('normals')
        
    def setup(self):
        
        num_nodes = self.options['num_nodes']
        normals = self.options['normals']
        k = self.options['k']
        

        #Inputs
        self.add_input('norm_vec',shape=(num_nodes,k,normals.shape[0],3))
        self.add_input('distance',shape=(num_nodes,k,normals.shape[0]))
    
        # outputs
        self.add_output('signedfun',shape=(k,1))


        # partials
        
        idx = np.tile(np.outer(np.arange(0,num_nodes*normals.shape[0]*k,k*normals.shape[0])\
                                    ,np.ones(normals.shape[0])).reshape(-1,1).T + np.tile(np.arange(normals.shape[0]),num_nodes),k)
        
        col_indices = idx + np.outer(np.arange(0,k*normals.shape[0],normals.shape[0]),np.ones(num_nodes*normals.shape[0])).reshape(-1,1).T        
        row_indices = np.outer(np.arange(k),np.ones(num_nodes*normals.shape[0]))
        
        idx2 = np.tile(np.outer(np.arange(0,num_nodes*normals.shape[0]*3*k,k*normals.shape[0]*3),np.ones(normals.shape[0]*3)).reshape(-1,1).T \
                                    + np.tile(np.arange(normals.shape[0]*3),num_nodes),k)
        col_indices2 = idx2 + np.outer(np.arange(0,k*normals.shape[0]*3,normals.shape[0]*3),np.ones(num_nodes*normals.shape[0]*3)).reshape(-1,1).T
        row_indices2 = np.outer(np.arange(k),np.ones(num_nodes*normals.shape[0]*3))
        
        self.declare_partials('signedfun', 'norm_vec',rows=row_indices2.flatten(),cols=col_indices2.flatten())
        self.declare_partials('signedfun', 'distance',rows=row_indices.flatten(),cols=col_indices.flatten())
        

        
        
    def compute(self,inputs,outputs):

        num_nodes = self.options['num_nodes']
        k = self.options['k']
        normals = self.options['normals']
        norm_vec = inputs['norm_vec']
        distance = inputs['distance']

        # vectorize the normals from mesh points
        normals_ = np.zeros((num_nodes,k,normals.shape[0],3))

        # Need to make sure the normals you get is facing inwards or outwards
        # Mine was facing outward originally, so I add the negative term here
        normals_[np.arange(num_nodes),:,:,:] = -normals
        self.normals_ = normals_
        
        # Compute the inner product
        # norm_vec is the vector of ctr backbone points
        # normals_ here is the normals you pre_computed from your mesh points 

        inner_product = np.einsum("ijkl,ijkl->ijk", norm_vec,normals_)
        self.inner_product = inner_product
        
        # The signed distance function, which is based on the 
        # signed distance function array
        f = (-1*inner_product) * (distance) # rank 2

        signedfun = np.sum(np.sum(f,axis=2),axis=0)
        outputs['signedfun'] = signedfun.reshape(-1,1) # (num_pts,1)
        # + outside
        # - inside
        # not normalized, proportional to distance
        # distance not needed to determine if inside/outside
        
        
    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        norm_vec = inputs['norm_vec']
        distance = inputs['distance']
        # cross_section = inputs['cross_section']
        normals = self.options['normals']
        normals_ = self.normals_
        inner_product = self.inner_product
       
        
        Pob1_pno = np.zeros((num_nodes,k,normals.shape[0],3))
        Pob1_pno[:,:,:,0] = -1 * normals_[:,:,:,0] * (distance) 
        Pob1_pno[:,:,:,1] = -1 * normals_[:,:,:,1] * (distance)
        Pob1_pno[:,:,:,2] = -1 * normals_[:,:,:,2] * (distance)

        tmp0 = np.vsplit(Pob1_pno,num_nodes)
        Po_pno = np.concatenate(tmp0,axis=2)
        Pob1_peu = np.zeros((num_nodes,k,normals.shape[0]))
        idx = np.arange(k)
        Pob1_peu[:,:,:] = -1*inner_product[:,:,:]
        tmp = np.vsplit(Pob1_peu,num_nodes)
        Po_peu = np.concatenate(tmp,axis=2)

        partials['signedfun','norm_vec'][:] = Po_pno.flatten()
        partials['signedfun','distance'][:] = Po_peu.flatten() 

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    from mesh import trianglemesh
    group = Group()
    num_nodes = 3
    k=1
    
    p = np.random.rand(10,3)
    normals = np.random.rand(10,3)
    comp = IndepVarComp()
   
    comp.add_output('norm_vec',val=np.random.random((num_nodes,k,p.shape[0],3)))
    comp.add_output('cross_section',val=np.random.random((num_nodes,k)))
    comp.add_output('distance',val=np.random.random((num_nodes,k,p.shape[0])))
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = SignedfunComp(num_nodes=num_nodes,normals=normals,k=k)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    