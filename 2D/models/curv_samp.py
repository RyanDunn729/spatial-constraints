from openmdao.api import ExplicitComponent
import numpy as np

class curv_sampling(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_cps', types=int)
        self.options.declare('num_pts', types=int)
        self.options.declare('dim', types=int)
        self.options.declare('scaling')
        self.options.declare('bases')

    def setup(self):
        num_pts = self.options['num_pts']
        num_cps = self.options['num_cps']
        self.add_input('phi_cps',shape=(num_cps,))
        self.add_output('dp_dxx',shape=(num_pts,))
        self.add_output('dp_dxy',shape=(num_pts,))
        self.add_output('dp_dyy',shape=(num_pts,))
        if self.options['dim'] == 3:
            self.add_output('dp_dxz',shape=(num_pts,))
            self.add_output('dp_dyz',shape=(num_pts,))
            self.add_output('dp_dzz',shape=(num_pts,))

    def compute(self, inputs, outputs):
        scaling = self.options['scaling']
        bases = self.options['bases']
        outputs['dp_dxx'] = scaling[0]*scaling[0]*bases[0].dot(inputs['phi_cps'])
        outputs['dp_dxy'] = scaling[0]*scaling[1]*bases[1].dot(inputs['phi_cps'])
        outputs['dp_dyy'] = scaling[1]*scaling[1]*bases[2].dot(inputs['phi_cps'])
        if self.options['dim'] == 3:
            outputs['dp_dxz'] = scaling[0]*scaling[2]*bases[3].dot(inputs['phi_cps'])
            outputs['dp_dyz'] = scaling[1]*scaling[2]*bases[4].dot(inputs['phi_cps'])
            outputs['dp_dzz'] = scaling[2]*scaling[2]*bases[5].dot(inputs['phi_cps'])

    def setup_partials(self):
        scaling = self.options['scaling']
        bases = self.options['bases']
        self.declare_partials(of='dp_dxx', wrt='phi_cps', val=scaling[0]*scaling[0]*bases[0])
        self.declare_partials(of='dp_dxy', wrt='phi_cps', val=scaling[0]*scaling[1]*bases[1])
        self.declare_partials(of='dp_dyy', wrt='phi_cps', val=scaling[1]*scaling[1]*bases[2])
        if self.options['dim'] == 3:
            self.declare_partials(of='dp_dxz', wrt='phi_cps', val=scaling[1]*scaling[2]*bases[3])
            self.declare_partials(of='dp_dyz', wrt='phi_cps', val=scaling[1]*scaling[2]*bases[4])
            self.declare_partials(of='dp_dzz', wrt='phi_cps', val=scaling[2]*scaling[2]*bases[5])
