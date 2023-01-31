from openmdao.api import ExplicitComponent
import numpy as np

class Fnorm(ExplicitComponent):

    def initialize(self):
        self.options.declare('dim', types=int)
        self.options.declare('num_pts', types=int)

    def setup(self):
        num_pts = self.options['num_pts']
        dim = self.options['dim']
        self.add_input('hessians',shape=(num_pts,dim,dim))
        self.add_output('Fnorm',shape=(num_pts,))
        
        row = np.repeat(np.arange(0,num_pts),dim**2)
        col = np.arange(0,num_pts*dim**2)

        self.declare_partials(of='Fnorm', wrt='hessians',
                              cols=col,
                              rows=row)

    def compute(self, inputs, outputs):
        outputs['Fnorm'] = np.linalg.norm(inputs['hessians'],axis=(1,2),ord='fro')

    def compute_partials(self,inputs,partials):
        dim = self.options['dim']
        norm = np.linalg.norm(inputs['hessians'],axis=(1,2),ord='fro')
        import time
        t1 = time.time()
        for i,hess in enumerate(inputs['hessians']):
            partials['Fnorm','hessians'][i*dim**2:(i+1)*dim**2] = hess.flatten()/norm[i]
        t2 = time.time()
        print(t2-t1)

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    import numpy as np

    num_pts = 100
    dim = 3

    group = Group()
    comp = Fnorm(num_pts=num_pts,dim=dim)
    group.add_subsystem('Fnorms', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group

    prob.setup()

    prob['hessians'] = np.random.rand(num_pts,dim,dim)
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)