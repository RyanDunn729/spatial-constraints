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
        outputs['phi_surf'] = bases[0].dot(inputs['phi_cps'])
        outputs['dpdx_surf'] = scaling[0]*bases[1].dot(inputs['phi_cps'])
        outputs['dpdy_surf'] = scaling[1]*bases[2].dot(inputs['phi_cps'])
        if self.options['dim'] == 3:
            outputs['dpdz_surf'] = scaling[2]*bases[3].dot(inputs['phi_cps'])

    def setup_partials(self):
        scaling = self.options['scaling']
        bases = self.options['bases']
        self.declare_partials(of='phi_surf', wrt='phi_cps',val=bases[0])
        self.declare_partials(of='dpdx_surf', wrt='phi_cps',val=scaling[0]*bases[1])
        self.declare_partials(of='dpdy_surf', wrt='phi_cps',val=scaling[1]*bases[2])
        if self.options['dim'] == 3:
            self.declare_partials(of='dpdz_surf', wrt='phi_cps',val=scaling[2]*bases[3])
