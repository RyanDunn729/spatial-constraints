from openmdao.api import ExplicitComponent
import numpy as np

class Fnorm(ExplicitComponent):

    def initialize(self):
        self.options.declare('dim', types=int)
        self.options.declare('num_pts', types=int)

    def setup(self):
        num_pts = self.options['num_pts']
        dim = self.options['dim']
        self.add_input('dp_dxx',shape=(num_pts,))
        self.add_input('dp_dxy',shape=(num_pts,))
        self.add_input('dp_dyy',shape=(num_pts,))
        if dim == 3:
            self.add_input('dp_dxz',shape=(num_pts,))
            self.add_input('dp_dyz',shape=(num_pts,))
            self.add_input('dp_dzz',shape=(num_pts,))
        self.add_output('Fnorm',shape=(1,))
        
        self.declare_partials(of='Fnorm', wrt='dp_dxx')
        self.declare_partials(of='Fnorm', wrt='dp_dxy')
        self.declare_partials(of='Fnorm', wrt='dp_dyy')
        if dim == 3:
            self.declare_partials(of='Fnorm', wrt='dp_dxz')
            self.declare_partials(of='Fnorm', wrt='dp_dyz')
            self.declare_partials(of='Fnorm', wrt='dp_dzz')

    def compute(self, inputs, outputs):
        num_pts = self.options['num_pts']
        dim = self.options['dim']
        dxx = inputs['dp_dxx']
        dxy = inputs['dp_dxy']
        dyy = inputs['dp_dyy']
        if dim == 3:
            dxz = inputs['dp_dxz']
            dyz = inputs['dp_dyz']
            dzz = inputs['dp_dzz']
        
        Fnorm = dxx**2 + 2*dxy**2 + dyy**2
        if dim == 3:
            Fnorm += 2*dxz**2 + 2*dyz**2 + dzz**2
        outputs['Fnorm'] = np.sum(Fnorm)

    def compute_partials(self,inputs,partials):
        num_pts = self.options['num_pts']
        dim = self.options['dim']
        dxx = inputs['dp_dxx']
        dxy = inputs['dp_dxy']
        dyy = inputs['dp_dyy']
        if dim == 3:
            dxz = inputs['dp_dxz']
            dyz = inputs['dp_dyz']
            dzz = inputs['dp_dzz']

        partials['Fnorm','dp_dxx'] = 2*dxx
        partials['Fnorm','dp_dxy'] = 4*dxy
        partials['Fnorm','dp_dyy'] = 2*dyy
        if dim == 3:
            partials['Fnorm','dp_dxz'] = 4*dxz
            partials['Fnorm','dp_dyz'] = 4*dyz
            partials['Fnorm','dp_dzz'] = 2*dzz

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    import numpy as np

    num_pts = 500
    dim = 3

    group = Group()
    comp = Fnorm(num_pts=num_pts,dim=dim)
    group.add_subsystem('Fnorms', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group

    prob.setup()

    prob['dp_dxx'] = np.random.rand(num_pts)
    prob['dp_dxy'] = np.random.rand(num_pts)
    prob['dp_dyy'] = np.random.rand(num_pts)
    if dim == 3:
        prob['dp_dxz'] = np.random.rand(num_pts)
        prob['dp_dyz'] = np.random.rand(num_pts)
        prob['dp_dzz'] = np.random.rand(num_pts)

    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)