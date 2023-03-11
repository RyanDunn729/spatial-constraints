from openmdao.api import ExplicitComponent
import numpy as np

class curv_sampling(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_cps', types=int)
        self.options.declare('num_pts', types=int)
        self.options.declare('dim', types=int)
        self.options.declare('scaling', types=np.ndarray)
        self.options.declare('bases')
        self.options.declare('bbox_diag', types=float)

    def setup(self):
        scaling = self.options['scaling']
        bases = self.options['bases']
        num_pts = self.options['num_pts']
        num_cps = self.options['num_cps']
        bbox_diag = self.options['bbox_diag']
        self.add_input('phi_cps',shape=(num_cps,))
        self.add_output('dp_dxx',shape=(num_pts,))
        self.add_output('dp_dxy',shape=(num_pts,))
        self.add_output('dp_dyy',shape=(num_pts,))
        if self.options['dim'] == 3:
            self.add_output('dp_dxz',shape=(num_pts,))
            self.add_output('dp_dyz',shape=(num_pts,))
            self.add_output('dp_dzz',shape=(num_pts,))

        self.declare_partials(of='dp_dxx', wrt='phi_cps', val=bbox_diag*bbox_diag*scaling[0]*scaling[0]*bases[0])
        self.declare_partials(of='dp_dxy', wrt='phi_cps', val=bbox_diag*bbox_diag*scaling[0]*scaling[1]*bases[1])
        self.declare_partials(of='dp_dyy', wrt='phi_cps', val=bbox_diag*bbox_diag*scaling[1]*scaling[1]*bases[2])
        if self.options['dim'] == 3:
            self.declare_partials(of='dp_dxz', wrt='phi_cps', val=bbox_diag*bbox_diag*scaling[0]*scaling[2]*bases[3])
            self.declare_partials(of='dp_dyz', wrt='phi_cps', val=bbox_diag*bbox_diag*scaling[1]*scaling[2]*bases[4])
            self.declare_partials(of='dp_dzz', wrt='phi_cps', val=bbox_diag*bbox_diag*scaling[2]*scaling[2]*bases[5])

    def compute(self, inputs, outputs):
        scaling = self.options['scaling']
        bases = self.options['bases']
        bbox_diag = self.options['bbox_diag']
        outputs['dp_dxx'] = bbox_diag*bbox_diag*scaling[0]*scaling[0]*bases[0].dot(inputs['phi_cps'])
        outputs['dp_dxy'] = bbox_diag*bbox_diag*scaling[0]*scaling[1]*bases[1].dot(inputs['phi_cps'])
        outputs['dp_dyy'] = bbox_diag*bbox_diag*scaling[1]*scaling[1]*bases[2].dot(inputs['phi_cps'])
        if self.options['dim'] == 3:
            outputs['dp_dxz'] = bbox_diag*bbox_diag*scaling[0]*scaling[2]*bases[3].dot(inputs['phi_cps'])
            outputs['dp_dyz'] = bbox_diag*bbox_diag*scaling[1]*scaling[2]*bases[4].dot(inputs['phi_cps'])
            outputs['dp_dzz'] = bbox_diag*bbox_diag*scaling[2]*scaling[2]*bases[5].dot(inputs['phi_cps'])

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    import numpy as np
    from scipy.sparse import csc_matrix

    num_cps = 25
    N = 20
    dim = 3

    scaling = np.random.rand(dim)
    bbox_diag = 11.24124
    Lp = 1e4
    Ln = 1e2
    Lr = 1e-2

    def gen_sp_matrix(*args):
        rand_matrix = np.random.rand(*args)
        rand_matrix[rand_matrix<0.8] = 0
        return csc_matrix(rand_matrix)

    hessian_bases = [gen_sp_matrix(N,num_cps) for _ in range(int(dim*(dim+1)/2))]

    group = Group()
    comp = curv_sampling(
        num_cps=num_cps,
        num_pts=N,
        dim=dim,
        scaling=scaling,
        bases=hessian_bases,
        bbox_diag=float(0.1237812)
    )
    group.add_subsystem('objective', comp, promotes = ['*'])
        
    prob = Problem()
    prob.model = group
    prob.setup()
    prob['phi_cps'] = np.random.rand(num_cps)
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)