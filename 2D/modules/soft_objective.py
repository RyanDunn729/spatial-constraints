from openmdao.api import ExplicitComponent
import numpy as np

class soft_objective(ExplicitComponent):

    def initialize(self):
        self.options.declare('dim', types=int)
        self.options.declare('num_samp', types=int)
        self.options.declare('num_surf', types=int)
        self.options.declare('normals', types=np.ndarray)

    def setup(self):
        num_surf = self.options['num_surf']
        num_samp = self.options['num_samp']
        dim = self.options['dim']
        
        if dim==2:
            self.add_input('dA',shape=(1,))
            self.add_input('A',shape=(1,))
        elif dim==3:
            self.add_input('dV',shape=(1,))
            self.add_input('V',shape=(1,))
        self.add_input('lambdas',shape=(3,))
        self.add_input('phi_surf',shape=(num_surf,))
        self.add_input('dpdx_surf',shape=(num_surf,))
        self.add_input('dpdy_surf',shape=(num_surf,))
        if dim==3:
            self.add_input('dpdz_surf',shape=(num_surf,))
        self.add_input('Fnorm',shape=(num_samp,))
        self.add_output('soft_objective',shape=(1,))

        self.declare_partials(of='soft_objective', wrt='phi_surf')
        self.declare_partials(of='soft_objective', wrt='dpdx_surf')
        self.declare_partials(of='soft_objective', wrt='dpdy_surf')
        if dim==3:
            self.declare_partials(of='soft_objective', wrt='dpdz_surf')
        self.declare_partials(of='soft_objective', wrt='Fnorm')

    def compute(self, inputs, outputs):
        num_surf = self.options['num_surf']
        num_samp = self.options['num_samp']
        normals = self.options['normals']
        dim = self.options['dim']
        L1 = inputs['lambdas'][0]
        L2 = inputs['lambdas'][1]
        L3 = inputs['lambdas'][2]

        o3 = L3*np.sum(inputs['phi_surf']**2)/num_surf
        o2 = L2*np.sum((inputs['dpdx_surf']+normals[:,0])**2)/num_surf
        o2 += L2*np.sum((inputs['dpdy_surf']+normals[:,1])**2)/num_surf
        if dim==3:
            o2 += L2*np.sum((inputs['dpdz_surf']+normals[:,2])**2)/num_surf
            o1 = L1*inputs['dV']/inputs['V']*np.sum(inputs['Fnorm']**2)/num_samp
        elif dim==2:
            o1 = L1*inputs['dA']/inputs['A']*np.sum(inputs['Fnorm']**2)/num_samp

        print('LrEr: ',o1)
        print('LnEn: ',o2)
        print('LpEp: ',o3)

        outputs['soft_objective'] = o1+o2+o3

    def compute_partials(self, inputs, partials):
        num_surf = self.options['num_surf']
        num_samp = self.options['num_samp']
        normals = self.options['normals']
        dim = self.options['dim']
        L1 = inputs['lambdas'][0]
        L2 = inputs['lambdas'][1]
        L3 = inputs['lambdas'][2]

        partials['soft_objective','phi_surf'] = 2*L3*inputs['phi_surf']/num_surf
        partials['soft_objective','dpdx_surf'] = 2*L2*(inputs['dpdx_surf']+normals[:,0])/num_surf
        partials['soft_objective','dpdy_surf'] = 2*L2*(inputs['dpdy_surf']+normals[:,1])/num_surf
        if dim==2:
            partials['soft_objective','Fnorm'] = 2*L1*inputs['dA']/inputs['A']*inputs['Fnorm']/num_samp
        elif dim==3:
            partials['soft_objective','dpdz_surf'] = 2*L2*(inputs['dpdz_surf']+normals[:,2])/num_surf
            partials['soft_objective','Fnorm'] = 2*L1*inputs['dV']/inputs['V']*inputs['Fnorm']/num_samp


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
    comp = soft_objective(num_samp=num_samp,num_surf=num_surf,dim=dim,normals=normals)
    group.add_subsystem('Soft_objective', comp, promotes = ['*'])
        
    prob = Problem()
    prob.model = group

    prob.setup()

    prob['lambdas'] = np.array([L1,L2,L3])
    prob['phi_surf'] = np.random.rand(num_surf)
    prob['dpdx_surf'] = np.random.rand(num_surf)
    prob['dpdy_surf'] = np.random.rand(num_surf)
    if dim==3:
        prob['dpdz_surf'] = np.random.rand(num_surf)
    prob['Fnorm'] = np.random.rand(num_samp)
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)