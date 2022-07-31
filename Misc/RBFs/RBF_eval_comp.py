import numpy as np
from openmdao.api import ExplicitComponent

class evaluate_RBF(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_RBFs', types=int)
        self.options.declare('num_pts', types=int)
        self.options.declare('r', types=float)
        self.options.declare('RBFs')

    def setup(self):
        num_pts = self.options['num_pts']
        self.add_input('pts', shape=(num_pts,3))
        self.add_output('phi', shape=(num_pts,))

    def setup_partials(self):
        num_pts = self.options['num_pts']
        row_ind = np.empty(3*num_pts)
        for i in range(num_pts):
            row_ind[3*i : 3*(i+1)] = i
        col_ind = np.arange(3*num_pts)
        self.declare_partials(of='phi', wrt='pts',
                              rows=row_ind,cols=col_ind)

    def compute(self, inputs, outputs):
        num_pts = self.options['num_pts']
        nodes = self.options['RBFs']
        r = self.options['r']
        pt = inputs['pts']

        for k,i_pt in enumerate(pt):
            norm = np.linalg.norm(i_pt-nodes, axis=1)
            outputs['phi'][k] = np.sum(np.exp(- (norm**2) / (r**2)))
    
    def compute_partials(self, inputs, partials):
        num_pts = self.options['num_pts']
        nodes = self.options['RBFs']
        r = self.options['r']
        pt = inputs['pts']

        deriv = np.empty((num_pts,3))
        for k,i_pt in enumerate(pt):
            diff = np.transpose(i_pt-nodes)
            norm = np.linalg.norm(diff, axis=0)
            var = 2/(-r**2) * np.exp( -(norm**2)/(r**2) )
            partials['phi','pts'][3*k:3*(k+1)] = np.sum(diff*var,axis=1).flatten()

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
    comp = evaluate_RBF(RBFs=nodes,num_RBFs=num_RBFs,num_pts=num_pts,r=r)
    group.add_subsystem('RBFs', comp, promotes = ['*'])
        
    prob = Problem()
    prob.model = group

    prob.setup()

    prob['pts'] = pt
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    # prob.check_partials(compact_print=True)