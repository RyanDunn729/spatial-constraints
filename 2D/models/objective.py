from openmdao.api import ExplicitComponent
import numpy as np

class Objective(ExplicitComponent):

    def initialize(self):
        self.options.declare('dim', types=int)
        self.options.declare('num_samp', types=int)
        self.options.declare('num_surf', types=int)
        self.options.declare('Lp', types=float)
        self.options.declare('Ln', types=float)
        self.options.declare('Lr', types=float)
        self.options.declare('normals', types=np.ndarray)
        self.options.declare('bbox_diag',types=float)
        self.options.declare('verbose',types=bool)

    def setup(self):
        num_surf = self.options['num_surf']
        num_samp = self.options['num_samp']
        dim = self.options['dim']
        Lp = self.options['Lp']
        Ln = self.options['Ln']
        Lr = self.options['Lr']
        normals = self.options['normals']
        bbox_diag = self.options['bbox_diag']
        verbose = self.options['verbose']
        
        self.add_input('phi_surf',shape=(num_surf,))
        self.add_input('dpdx_surf',shape=(num_surf,))
        self.add_input('dpdy_surf',shape=(num_surf,))
        if dim==3:
            self.add_input('dpdz_surf',shape=(num_surf,))
        self.add_input('Fnorm',shape=(1,))
        self.add_output('objective',shape=(1,))

        self.declare_partials(of='objective', wrt='phi_surf')
        self.declare_partials(of='objective', wrt='dpdx_surf')
        self.declare_partials(of='objective', wrt='dpdy_surf')
        if dim==3:
            self.declare_partials(of='objective', wrt='dpdz_surf')
        self.declare_partials(of='objective', wrt='Fnorm')

    def compute(self, inputs, outputs):
        num_surf = self.options['num_surf']
        num_samp = self.options['num_samp']
        dim = self.options['dim']
        Lp = self.options['Lp']
        Ln = self.options['Ln']
        Lr = self.options['Lr']
        normals = self.options['normals']
        bbox_diag = self.options['bbox_diag']
        verbose = self.options['verbose']

        nx = normals[:,0]/bbox_diag
        ny = normals[:,1]/bbox_diag
        if dim == 3:
            nz = normals[:,2]/bbox_diag

        Er = inputs['Fnorm']/num_samp/num_samp
        Ep = np.sum(inputs['phi_surf']**2)/num_surf
        En = np.sum((inputs['dpdx_surf']+nx)**2)/num_surf
        En += np.sum((inputs['dpdy_surf']+ny)**2)/num_surf
        if dim==3:
            En += np.sum((inputs['dpdz_surf']+nz)**2)/num_surf

        if verbose:
            print('Er: ',Er)
            print('En: ',En)
            print('Ep: ',Ep)

        outputs['objective'] = Lp*Ep + Ln*En + Lr*Er

    def compute_partials(self, inputs, partials):
        num_surf = self.options['num_surf']
        num_samp = self.options['num_samp']
        dim = self.options['dim']
        Lp = self.options['Lp']
        Ln = self.options['Ln']
        Lr = self.options['Lr']
        normals = self.options['normals']
        bbox_diag = self.options['bbox_diag']
        verbose = self.options['verbose']

        nx = normals[:,0]/bbox_diag
        ny = normals[:,1]/bbox_diag
        if dim == 3:
            nz = normals[:,2]/bbox_diag

        partials['objective','phi_surf'] = 2*Lp*inputs['phi_surf']/num_surf
        partials['objective','Fnorm'] = Lr/num_samp/num_samp
        partials['objective','dpdx_surf'] = 2*Ln*(inputs['dpdx_surf']+nx)/num_surf
        partials['objective','dpdy_surf'] = 2*Ln*(inputs['dpdy_surf']+ny)/num_surf
        if dim==3:
            partials['objective','dpdz_surf'] = 2*Ln*(inputs['dpdz_surf']+nz)/num_surf


if __name__ == '__main__':    
    from openmdao.api import Problem, Group
    import numpy as np

    dim = 3
    num_surf = 10
    num_samp = 20

    L1 = 1.
    L2 = 4.
    L3 = 3.

    normals = np.random.rand(num_surf,3)

    group = Group()
    comp = Objective(
        num_samp=num_samp,
        num_surf=num_surf,
        dim=dim,
        Lp=1.12,
        Ln=0.5124,
        Lr=0.5124587,
        normals=normals,
        bbox_diag=0.12378,
        verbose=False,
    )
    group.add_subsystem('objective', comp, promotes = ['*'])
        
    prob = Problem()
    prob.model = group

    prob.setup()

    prob['phi_surf'] = np.random.rand(num_surf)
    prob['dpdx_surf'] = np.random.rand(num_surf)
    prob['dpdy_surf'] = np.random.rand(num_surf)
    if dim==3:
        prob['dpdz_surf'] = np.random.rand(num_surf)
    prob['Fnorm'] = 0.124124
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)