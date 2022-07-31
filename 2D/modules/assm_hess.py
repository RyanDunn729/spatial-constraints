from openmdao.api import ExplicitComponent
import numpy as np

class assm_hess(ExplicitComponent):

    def initialize(self):
        self.options.declare('dim', types=int)
        self.options.declare('num_pts', types=int)

    def setup(self):
        num_pts = self.options['num_pts']
        dim = self.options['dim']
        self.add_output('hessians',shape=(num_pts,dim,dim))
        self.add_input('dp_dxx',shape=(num_pts,))
        self.add_input('dp_dxy',shape=(num_pts,))
        self.add_input('dp_dyy',shape=(num_pts,))
        if self.options['dim'] == 3:
            self.add_input('dp_duw',shape=(num_pts,))
            self.add_input('dp_dyz',shape=(num_pts,))
            self.add_input('dp_dzz',shape=(num_pts,))
        
        self.declare_partials(of='hessians', wrt='dp_dxx',
                              cols=np.arange(0,num_pts),
                              rows=dim**2*np.arange(0,num_pts),
                              val=np.ones(num_pts))
        
        rows = np.stack((np.arange(0,num_pts)*dim**2+1,
                         np.arange(0,num_pts)*dim**2+dim),
                         axis=1).flatten()
        self.declare_partials(of='hessians', wrt='dp_dxy',
                              cols=np.repeat(np.arange(0,num_pts),2),
                              rows=rows,
                              val=np.ones(2*num_pts))
        self.declare_partials(of='hessians', wrt='dp_dyy',
                              cols=np.arange(0,num_pts),
                              rows=dim**2*(np.arange(0,num_pts))+dim+1,
                              val=np.ones(num_pts))
        if self.options['dim'] == 3:
            rows = np.stack((np.arange(0,num_pts)*dim**2+2,
                             np.arange(0,num_pts)*dim**2+2*dim),
                             axis=1).flatten()
            self.declare_partials(of='hessians', wrt='dp_duw',
                                  cols=np.repeat(np.arange(0,num_pts),2),
                                  rows=rows,
                                  val=np.ones(2*num_pts))
            rows = np.stack((np.arange(0,num_pts)*dim**2+2*dim-1,
                             np.arange(0,num_pts)*dim**2+2*dim+1),
                             axis=1).flatten()
            self.declare_partials(of='hessians', wrt='dp_dyz',
                                  cols=np.repeat(np.arange(0,num_pts),2),
                                  rows=rows,
                                  val=np.ones(2*num_pts))
            self.declare_partials(of='hessians', wrt='dp_dzz',
                                  cols=np.arange(0,num_pts),
                                  rows=dim**2*(np.arange(1,num_pts+1))-1,
                                  val=np.ones(num_pts))

    def compute(self, inputs, outputs):
        num_pts = self.options['num_pts']
        dim = self.options['dim']
        hess = np.zeros((num_pts,dim,dim))
        hess[:,0,0] = inputs['dp_dxx']
        hess[:,1,0] = inputs['dp_dxy']
        hess[:,0,1] = inputs['dp_dxy']
        hess[:,1,1] = inputs['dp_dyy']
        if self.options['dim'] == 3:
            hess[:,0,2] = inputs['dp_duw']
            hess[:,2,0] = inputs['dp_duw']
            hess[:,1,2] = inputs['dp_dyz']
            hess[:,2,1] = inputs['dp_dyz']
            hess[:,2,2] = inputs['dp_dzz']
        outputs['hessians'] = hess

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    import numpy as np

    num_pts = 5
    dim = 2

    group = Group()
    comp = assm_hess(num_pts=num_pts,dim=dim)
    group.add_subsystem('Eigenvalues', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group

    prob.setup()

    prob['dp_dxx'] = np.random.rand(num_pts)
    prob['dp_dxy'] = np.random.rand(num_pts)
    prob['dp_dyy'] = np.random.rand(num_pts)
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)