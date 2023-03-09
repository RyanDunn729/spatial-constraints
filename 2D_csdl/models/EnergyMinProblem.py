import scipy.sparse as sps
import numpy as np
import csdl


class EnergyMinProblem(csdl.Model):

    def initialize(self):
        self.parameters.declare('num_cps', types=int)
        self.parameters.declare('N_gamma', types=int)
        self.parameters.declare('N',       types=int)
        self.parameters.declare('dim',     types=int)
        self.parameters.declare('scaling', types=np.ndarray)
        self.parameters.declare('dV',      types=float)
        self.parameters.declare('Lp',      types=float)
        self.parameters.declare('Ln',      types=float)
        self.parameters.declare('Lr',      types=float)
        self.parameters.declare('normals', types=np.ndarray)
        self.parameters.declare('scalar_basis',  types=sps.spmatrix)
        self.parameters.declare('gradient_bases',types=list)
        self.parameters.declare('hessian_bases', types=list)

    def define(self):
        # Parameters
        gradient_bases = self.parameters['gradient_bases']
        hessian_bases = self.parameters['hessian_bases']
        scalar_basis = self.parameters['scalar_basis']
        scaling = self.parameters['scaling']
        num_cps = self.parameters['num_cps']
        N_gamma = self.parameters['N_gamma']
        normals = self.parameters['normals']
        dim = self.parameters['dim'] 
        Lp = self.parameters['Lp']
        Ln = self.parameters['Ln']
        Lr = self.parameters['Lr']
        dV = self.parameters['dV']
        N = self.parameters['N']
        # Design Variables
        phi_cps = self.declare_variable('phi_cps',shape=(num_cps,))
        # Importing basis matrices
        basis_000 = scalar_basis
        basis_100 = gradient_bases[0]
        basis_010 = gradient_bases[1]
        basis_200 = hessian_bases[0]
        basis_110 = hessian_bases[1]
        basis_020 = hessian_bases[2]
        if dim == 3:
            basis_001 = gradient_bases[3]
            basis_101 = hessian_bases[4]
            basis_011 = hessian_bases[5]
            basis_002 = hessian_bases[6]  
        # Sampling the Bsplines
        phi_surf = csdl.matvec(basis_000,phi_cps)
        dx = scaling[0]*csdl.matvec(basis_100,phi_cps)
        dy = scaling[1]*csdl.matvec(basis_010,phi_cps)
        dxx = scaling[0]*scaling[0]*csdl.matvec(basis_200,phi_cps)
        dxy = scaling[0]*scaling[1]*csdl.matvec(basis_110,phi_cps)
        dyy = scaling[1]*scaling[1]*csdl.matvec(basis_020,phi_cps)
        if dim == 3:
            dz = scaling[2]*csdl.matvec(basis_001,phi_cps)
            dxz = scaling[0]*scaling[2]*csdl.matvec(basis_101,phi_cps)
            dyz = scaling[1]*scaling[2]*csdl.matvec(basis_011,phi_cps)
            dzz = scaling[2]*scaling[2]*csdl.matvec(basis_002,phi_cps)
        # CSDL reshaping at its finest
        nx = normals[:,0]
        ny = normals[:,1]
        if dim == 3:
            nz = normals[:,2]
        # Assembling energy terms
        Ep = 1/N_gamma * csdl.sum(phi_surf**2)
        if dim == 2:
            En = 1/N_gamma * csdl.sum( ((dx+nx)**2 + (dy+ny)**2) )
            Er = 1/N * csdl.sum( dxx**2 + 2*dxy**2 + dyy**2)
        if dim == 3:
            En = 1/N_gamma * csdl.sum( ((dx+nx)**2 + (dy+ny)**2 + (dz+nz)**2) )
            Er = 1/N * csdl.sum( dxx**2 + 2*dxy**2 + dyy**2 + 2*dxz**2 + 2*dyz**2 + dzz**2)
        # Optional Printing
        self.print_var(Ep)
        self.print_var(En)
        self.print_var(Er)
        # Objective function
        f = Lp*Ep + Ln*En + Lr*Er
        self.register_output("objective",f)
if __name__ == '__main__':
    from python_csdl_backend import Simulator
    import numpy as np
    from scipy.sparse import csc_matrix

    num_cps = 25
    N_gamma = 5
    N = 20
    dim = 3

    scaling = np.random.rand(dim)
    dV = 0.83493758
    Lp = 1e4
    Ln = 1e2
    Lr = 1e-2

    def gen_sp_matrix(*args):
        rand_matrix = np.random.rand(*args)
        rand_matrix[rand_matrix<0.8] = 0
        return csc_matrix(rand_matrix)
    sp_matrix = gen_sp_matrix(N_gamma,num_cps)

    scalar_basis = gen_sp_matrix(N_gamma,num_cps)
    gradient_bases = [gen_sp_matrix(N_gamma,num_cps) for _ in range(dim)]
    hessian_bases = [gen_sp_matrix(N,num_cps) for _ in range(int(dim*(dim+1)/2))]
    normals = np.random.rand(N_gamma,dim)
    sim = Simulator(EnergyMinProblem(
        dim=dim,
        scaling=scaling,
        num_cps=num_cps,
        N_gamma=N_gamma,
        N=N,
        Lp=Lp,
        Ln=Ln,
        Lr=Lr,
        dV=dV,
        scalar_basis=scalar_basis,
        gradient_bases=gradient_bases,
        hessian_bases=hessian_bases,
        normals=normals
    ))
    sim['phi_cps'] = np.random.rand(num_cps)
    sim.run()

    sim.check_totals(of="objective",wrt="phi_cps",compact_print=True)