import numpy as np
from openmdao.api import ExplicitComponent

class manipulate_RBF(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_RBFs', types=int)
        self.options.declare('num_pts', types=int)
        self.options.declare('r', types=float)
        self.options.declare('pts')

    def setup(self):
        num_RBFs = self.options['num_RBFs']
        self.add_input('RBFs', shape=(num_RBFs,3))
        self.add_output('phi', shape=(num_pts,))

    def setup_partials(self):
        self.declare_partials(of='phi', wrt='RBFs')

    def compute(self, inputs, outputs):
        num_RBFs = self.options['num_RBFs']
        num_pts = self.options['num_pts']
        pt = self.options['pts']
        r = self.options['r']
        nodes = inputs['RBFs']

        for k,i_pt in enumerate(pt):
            norm = np.linalg.norm(i_pt-nodes, axis=1)
            outputs['phi'][k] = np.sum(np.exp(- (norm**2) / (r**2)))
    
    def compute_partials(self, inputs, partials):
        num_RBFs = self.options['num_RBFs']
        num_pts = self.options['num_pts']
        pt = self.options['pts']
        r = self.options['r']
        nodes = inputs['RBFs']

        for k,i_pt in enumerate(pt):
            diff = np.transpose(i_pt-nodes)
            norm = np.linalg.norm(diff,axis=0)
            var = 2/(r**2) * np.exp( -(norm**2)/(r**2) )
            partials['phi','RBFs'][k] = (diff*var).transpose().flatten()

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    import numpy as np

    r = 1.5

    num_RBFs = 1000
    ang = np.linspace(0,2*np.pi,num_RBFs)
    nodes = np.column_stack((3*np.cos(ang),np.sin(ang),ang))

    res = 5
    num_pts = res**3
    rng = 1
    pt = np.empty((num_pts,3))
    pt[:,0] = np.einsum('i,j,k->ijk', np.linspace(-rng,rng,res), np.ones(res), np.ones(res)).flatten()
    pt[:,1] = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(-rng,rng,res), np.ones(res)).flatten()
    pt[:,2] = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res), np.linspace(-rng,rng,res)).flatten()

    group = Group()
    comp = manipulate_RBF(pts=pt,num_RBFs=num_RBFs,num_pts=num_pts,r=r)
    group.add_subsystem('RBFs', comp, promotes = ['*'])
        
    prob = Problem()
    prob.model = group

    prob.setup()

    prob['RBFs'] = nodes
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    # prob.check_partials(compact_print=True)