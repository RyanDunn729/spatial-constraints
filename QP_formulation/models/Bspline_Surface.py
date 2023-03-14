import numpy as np
import scipy.sparse as sps

class BSplineSurface:
    def __init__(self, name, order_u, order_v, knots_u, knots_v, shape):
        self.name = name
        self.order_u = order_u
        self.order_v = order_v
        self.knots_u = knots_u
        self.knots_v = knots_v
        self.shape_u = int(shape[0])
        self.shape_v = int(shape[1])
        self.num_control_points = int(shape[0]*shape[1])
        
    def get_basis_matrix(self, u_vec, v_vec, du, dv):
        data = np.zeros(len(u_vec) * self.order_u * self.order_v)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        data, row_indices, col_indices = self.get_basis_surface_matrix(
            du, dv, u_vec, v_vec,
            data, row_indices, col_indices
            )
            
        basis = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis
        
    def get_basis_surface_matrix(self, du, dv, u_vec, v_vec, data, row_indices, col_indices):        
        i_nz = 0
        
        if du == 0:
            get_basis_u = self.get_basis0
        elif du == 1:
            get_basis_u = self.get_basis1
        elif du == 2:
            get_basis_u = self.get_basis2

        if dv == 0:
            get_basis_v = self.get_basis0
        elif dv == 1:
            get_basis_v = self.get_basis1
        elif dv == 2:
            get_basis_v = self.get_basis2
        
        for i_pt in range(len(u_vec)):
            
            i_start_u, basis_u = get_basis_u(self.order_u, self.shape_u, u_vec[i_pt], self.knots_u)
            i_start_v, basis_v = get_basis_v(self.order_v, self.shape_v, v_vec[i_pt], self.knots_v)
            
            for i_order_u in range(self.order_u):
                for i_order_v in range(self.order_v):
                    data[i_nz] = basis_u[i_order_u] * basis_v[i_order_v]
                    row_indices[i_nz] = i_pt
                    col_indices[i_nz] = self.shape_v * (i_start_u + i_order_u) + (i_start_v + i_order_v)

                    i_nz += 1
                    
        return data, row_indices, col_indices
    
    def get_basis0(self, order, num_control_points, u, knot_vector):
        basis0 = np.zeros(order)
        i_start = -1

        # Find the knot interval
        for i in range(order - 1, num_control_points):
            if (knot_vector[i] <= u) and (u < knot_vector[i + 1]):
                i_start = i - order + 1

        # Initialize the basis0 to (0., ..., 0., 1.)
        for i in range(order - 1):
            basis0[i] = 0.

        basis0[order - 1] = 1.

        if abs(u - knot_vector[num_control_points]) < 1e-14:
            i_start = num_control_points - order

        for i in range(1, order):
            j1 = order - i
            j2 = order
            n = i_start + j1

            if knot_vector[n + i] != knot_vector[n]:
                basis0[j1 - 1] = ( knot_vector[n + i] - u ) / ( knot_vector[n + i] - knot_vector[n] ) * basis0[j1]
            else:
                basis0[j1 - 1] = 0.

            for j in range(j1, j2 - 1):
                n = i_start + j + 1
                if knot_vector[n + i - 1] != knot_vector[n - 1]:
                    basis0[j] = ( u - knot_vector[n - 1] ) / ( knot_vector[n + i - 1] - knot_vector[n - 1] ) * basis0[j]
                else:
                    basis0[j] = 0.
                if knot_vector[n + i] != knot_vector[n]:
                    basis0[j] += ( knot_vector[n + i] - u ) / ( knot_vector[n + i] - knot_vector[n] ) * basis0[j + 1]

            n = i_start + j2

            if knot_vector[n + i - 1] != knot_vector[n - 1]:
                basis0[j2 - 1] = ( u - knot_vector[n - 1] ) / ( knot_vector[n + i - 1] - knot_vector[n - 1] ) * basis0[j2 - 1]
            else:
                basis0[j2 - 1] = 0.

        return i_start, basis0
        
        
    def get_basis1(self, order, num_control_points, u, knot_vector):
        basis0 = np.zeros(order)
        basis1 = np.zeros(order)
        i_start = -1

        # Find the knot interval
        for i in range(order - 1, num_control_points):
            if (knot_vector[i] <= u) and (u < knot_vector[i + 1]):
                i_start = i - order + 1

        # Initialize the basis0 to (0., ..., 0., 1.)
        for i in range(order - 1):
            basis0[i] = 0.

        basis0[order - 1] = 1.

        # If parameter is at the maximum of the knot vector, set the i_start appropriately
        if abs(u - knot_vector[num_control_points]) < 1e-14:
            i_start = num_control_points - order

        for i in range(order):
            basis1[i] = 0.

        # Recursion loop over the order index
        for i in range(1, order):
            j1 = order - i
            j2 = order

            for j in range(j1 - 1, j2):
                n = i_start + j + 1
                if knot_vector[n + i - 1] != knot_vector[n - 1]:
                    den = knot_vector[n + i - 1] - knot_vector[n - 1]
                    b0_a = ( u - knot_vector[n - 1] ) / den * basis0[j]
                    b1_a = ( basis0[j] + ( u - knot_vector[n - 1] ) * basis1[j] ) / den
                else:
                    b0_a = 0.
                    b1_a = 0.
                if j != j2 - 1 and  knot_vector[n + i] != knot_vector[n]:
                    den = knot_vector[n + i] - knot_vector[n]
                    b0_b = ( knot_vector[n + i] - u ) / den * basis0[j + 1]
                    b1_b = ( ( knot_vector[n + i] - u ) * basis1[j + 1] - basis0[j + 1] ) / den
                else:
                    b0_b = 0.
                    b1_b = 0.

                basis0[j] = b0_a + b0_b
                basis1[j] = b1_a + b1_b

        return i_start, basis1

    def get_basis2(self, order, num_control_points, u, knot_vector):
        basis0 = np.zeros(order)
        basis1 = np.zeros(order)
        basis2 = np.zeros(order)
        i_start = -1

        # Find the knot interval
        for i in range(order - 1, num_control_points):
            if (knot_vector[i] <= u) and (u < knot_vector[i + 1]):
                i_start = i - order + 1

        # Initialize the basis0 to (0., ..., 0., 1.)
        for i in range(order - 1):
            basis0[i] = 0.

        basis0[order - 1] = 1.

        # If parameter is at the maximum of the knot vector, set the i_start appropriately
        if abs(u - knot_vector[num_control_points]) < 1e-14:
            i_start = num_control_points - order

        for i in range(order):
            basis1[i] = 0.
            basis2[i] = 0.

        # Recursion loop over the order index
        for i in range(1, order):
            j1 = order - i
            j2 = order

            for j in range(j1 - 1, j2):
                n = i_start + j + 1
                if knot_vector[n + i - 1] != knot_vector[n - 1]:
                    den = knot_vector[n + i - 1] - knot_vector[n - 1]
                    b0_a = ( u - knot_vector[n - 1] ) / den * basis0[j]
                    b1_a = ( basis0[j] + ( u - knot_vector[n - 1] ) * basis1[j] ) / den
                    b2_a = ( 2 * basis1[j] + ( u - knot_vector[n - 1] ) * basis2[j] ) / den
                else:
                    b0_a = 0.
                    b1_a = 0.
                    b2_a = 0.
                if j != j2 - 1 and  knot_vector[n + i] != knot_vector[n]:
                    den = knot_vector[n + i] - knot_vector[n]
                    b0_b = ( knot_vector[n + i] - u ) / den * basis0[j + 1]
                    b1_b = ( ( knot_vector[n + i] - u ) * basis1[j + 1] - basis0[j + 1] ) / den
                    b2_b = ( ( knot_vector[n + i] - u ) * basis2[j + 1] - 2 * basis1[j + 1] ) / den
                else:
                    b0_b = 0.
                    b1_b = 0.
                    b2_b = 0.

                basis0[j] = b0_a + b0_b
                basis1[j] = b1_a + b1_b
                if i > 1:
                    basis2[j] = b2_a + b2_b

        return i_start, basis2

def main():
    import time
    import matplotlib.pyplot as plt
    
    def std_uniform_knot_vec(num_cps,order):
        knot_vector = np.zeros(num_cps + order)
        for i in range(num_cps + order):
            knot_vector[i] = (i - order + 1) / (num_cps - order + 1)
        return knot_vector
    
    order = 4
    num_cps = [100,30]
    
    kv_u = std_uniform_knot_vec(num_cps[0],order)
    kv_v = std_uniform_knot_vec(num_cps[1],order)
    
    cps = np.zeros((np.product(num_cps), 3))
    cps[:, 0] = np.einsum('i,j->ij', np.linspace(0,1,num_cps[0]), np.ones(num_cps[1])).flatten()
    cps[:, 1] = np.einsum('i,j->ij', np.ones(num_cps[0]), np.linspace(0,1,num_cps[1])).flatten()
    cps[:, 2] = np.random.rand(np.product(num_cps))
    
    Surface = BSplineSurface('name',order,order,kv_u,kv_v,num_cps)
    
    res = 15
    time_set = np.zeros(res)
    time_data = np.logspace(1,5,res,dtype=int)
    for i,num_pts in enumerate(time_data):
        u_vec = np.linspace(0,1,num_pts)
        v_vec = np.linspace(0,1,num_pts)
        t1 = time.perf_counter()
        basis = Surface.get_basis_matrix(u_vec,v_vec,0,0)
        p = basis.dot(cps[:,2])
        t2 = time.perf_counter()
        time_set[i] = t2-t1
        print(num_pts,t2-t1)
    
    plt.loglog(time_data,time_set)
    plt.show()
    print(t2-t1)

def main2():
    import matplotlib.pyplot as plt
    
    def std_uniform_knot_vec(num_cps,order):
        knot_vector = np.zeros(num_cps + order)
        for i in range(num_cps + order):
            knot_vector[i] = (i - order + 1) / (num_cps - order + 1)
        return knot_vector
    
    order = 4
    num_cps = [10,15]
    
    kv_u = std_uniform_knot_vec(num_cps[0],order)
    kv_v = std_uniform_knot_vec(num_cps[1],order)
    
    cps = np.zeros((np.product(num_cps), 3))
    cps[:, 0] = np.einsum('i,j->ij', np.linspace(0,1,num_cps[0]), np.ones(num_cps[1])).flatten()
    cps[:, 1] = np.einsum('i,j->ij', np.ones(num_cps[0]), np.linspace(0,1,num_cps[1])).flatten()
    cps[:, 2] = np.random.rand(np.product(num_cps))
    
    Surface = BSplineSurface('name',order,order,kv_u,kv_v,num_cps)
    u_vec,v_vec = np.meshgrid(
        np.linspace(0,1,12),
        np.linspace(0,1,12),
    )
    basis = Surface.get_basis_matrix(u_vec,2)



if __name__ == '__main__':
    # main()
    main2()