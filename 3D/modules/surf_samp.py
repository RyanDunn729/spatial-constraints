from openmdao.api import ExplicitComponent
import numpy as np

# Component evaluates the surface gradient and value

class surf_sampling(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_cps', types=int)
        self.options.declare('num_pts', types=int)
        self.options.declare('dim', types=int)
        self.options.declare('scaling')
        self.options.declare('bases')
        self.options.declare('bbox_diag',types=float)

    def setup(self):
        num_pts = self.options['num_pts']
        num_cps = self.options['num_cps']

        self.add_input('phi_cps',shape=(num_cps,))
        self.add_output('phi_surf',shape=(num_pts,))
        self.add_output('dpdx_surf',shape=(num_pts,))
        self.add_output('dpdy_surf',shape=(num_pts,))
        if self.options['dim'] == 3:
            self.add_output('dpdz_surf',shape=(num_pts,))

    def compute(self, inputs, outputs):
        scaling = self.options['scaling']
        bases = self.options['bases']
        bbox_diag = self.options['bbox_diag']

        outputs['phi_surf'] = bbox_diag*bases[0].dot(inputs['phi_cps'])
        outputs['dpdx_surf'] = bbox_diag*scaling[0]*bases[1].dot(inputs['phi_cps'])
        outputs['dpdy_surf'] = bbox_diag*scaling[1]*bases[2].dot(inputs['phi_cps'])
        if self.options['dim'] == 3:
            outputs['dpdz_surf'] = bbox_diag*scaling[2]*bases[3].dot(inputs['phi_cps'])

    def setup_partials(self):
        scaling = self.options['scaling']
        bases = self.options['bases']
        bbox_diag = self.options['bbox_diag']

        self.declare_partials(of='phi_surf', wrt='phi_cps',val=bbox_diag*bases[0])
        self.declare_partials(of='dpdx_surf', wrt='phi_cps',val=bbox_diag*scaling[0]*bases[1])
        self.declare_partials(of='dpdy_surf', wrt='phi_cps',val=bbox_diag*scaling[1]*bases[2])
        if self.options['dim'] == 3:
            self.declare_partials(of='dpdz_surf', wrt='phi_cps',val=bbox_diag*scaling[2]*bases[3])

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

    scalar_gradient_bases = [gen_sp_matrix(N,num_cps) for _ in range(int(1+dim))]

    group = Group()
    comp = surf_sampling(
        num_cps=num_cps,
        num_pts=N,
        dim=dim,
        scaling=scaling,
        bases=scalar_gradient_bases,
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