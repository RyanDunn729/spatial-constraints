import numpy as np
from scipy.spatial import KDTree

def Hicken_eval(pts,dataset,norm_vec,k,rho,curv=None):
    distances,indices = dataset.query(pts,k=k)
    d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
    exp = np.exp(-rho*d_norm)
    phi_lin = np.empty(len(pts))
    if curv is not None:
        dist_quad = np.empty((len(pts),k))
    for i,(i_pt,ind) in enumerate(zip(pts,indices)):
        dx = dataset.data[ind] - i_pt
        phi_lin[i] = np.dot(np.einsum('ij,ij->i',dx,norm_vec[ind]),exp[i]/np.sum(exp[i]))
        if curv is None:
            phi_quad = None
        else:
            m = np.tile(np.eye(2),(k,1,1)) - np.einsum('ij,ik->ijk',norm_vec[ind],norm_vec[ind])
            q = np.einsum('ik,ijk,ij->i',dx,m,dx)
            quadratic = curv[ind] * q /2
            dist_quad[i,:] += quadratic
    if curv is not None:
        phi_quad = np.einsum('ij,ij->i',dist_quad,exp)/np.sum(exp,axis=1)
    return phi_lin,phi_quad

def Exact_eval(pts,exact):
    dataset = KDTree(exact)
    distances,_ = dataset.query(pts,k=1)
    return distances

def Hicken_deriv_eval(pts,dataset,norm_vec,k,rho):
    distances,indices = dataset.query(pts,k=k)
    di = dataset.data[indices] - pts.reshape(num_pts,1,3)
    check = 2*np.heaviside(np.einsum('ijk,ijk->ij',di,norm_vec[indices]),1) - 1
    sign = 2*np.heaviside(np.sum(check,axis=1),1) - 1

    d_norm = (distances.T - distances[:,0]).T
    exp = np.exp(-rho*d_norm)

    dhi = np.empty((len(pts),3))
    dlow = np.empty((len(pts),3))
    for i,(i_pt,ind) in enumerate(zip(pts,indices)):
        k_pts = dataset.data[ind]

        dx = np.transpose((i_pt-k_pts).T/(distances[i] + 1e-20)) # Avoid dividing by zero when distance = 0

        hi_terms = np.empty((k,3))
        dexp = np.empty((k,3))
        dexp = dx-np.repeat(dx[0],k).reshape(3,k).T
        hi_terms = dx - rho*np.einsum('i,ij->ij',distances[i],dexp)
        dhi[i] = np.einsum('ij,i->j',hi_terms,exp[i])
        dlow[i] = np.einsum('ij,i->j',-rho*dexp,exp[i])

    low = np.sum(exp,axis=1)
    hi = np.einsum('ij,ij->i',distances,exp)
    
    # Quotient Rule (OLD)
    deriv = np.einsum('i,ij->ij', 1/low**2, (np.einsum('i,ij->ij',sign*low,dhi) - np.einsum('i,ij->ij',sign*hi,dlow)))
    return deriv

def KS_eval(pts,KDTree,norm_vec,k,rho):
    distances,indices = KDTree.query(pts,k=k)
    d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
    exp = np.exp(-rho*d_norm)
    Dx = KDTree.data[indices] - np.reshape(pts,(pts.shape[0],1,pts.shape[1]))
    phi = np.einsum('ijk,ijk,ij,i->i',Dx,norm_vec[indices],exp,1/np.sum(exp,axis=1))
    return phi

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    
    exact = np.random.rand(10000,3)
    dataset = KDTree(exact)
    norm_vec = np.random.rand(10000,3)

    k = 4
    rho = 10

    res = 200
    time_set = np.zeros(res)
    deriv_time_set = np.zeros(res)
    time_data = np.logspace(1,3,res,dtype=int)
    for i,num_pts in enumerate(time_data):
        i_pts = np.random.rand(num_pts,3)
        t1 = time.perf_counter()
        phi = KS_eval(i_pts,dataset,norm_vec,k,rho)
        t2 = time.perf_counter()
        time_set[i] = t2-t1
        t1 = time.perf_counter()
        deriv = Hicken_deriv_eval(i_pts,dataset,norm_vec,k,rho)
        t2 = time.perf_counter()
        deriv_time_set[i] = t2-t1
        print(num_pts,time_set[i],deriv_time_set[i])
    plt.loglog(time_data,time_set,label='Evaluation')
    plt.loglog(time_data,deriv_time_set,label='Derivative Evaluation')
    plt.legend()
    plt.xlabel('Number of evaluations')
    plt.ylabel('CPU Time')
    plt.show()